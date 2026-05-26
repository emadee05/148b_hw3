"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
        --method lora --rank 8 --alpha 16 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from basics.vit import ViT
from basics.lora import apply_lora_to_attention

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANKS = [1, 2, 4, 8, 16, 32, 64]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--method",
        choices=["linear_probe", "lora", "full_ft"],
        required=True,
    )
    p.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    p.add_argument(
        "--pretrained",
        type=Path,
        required=True,
        help="Path to CLIP-pretrained ViT checkpoint from §3",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def cfg_get(cfg: dict, key: str, default):
    if key in cfg:
        return cfg[key]
    for v in cfg.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return default


def build_resisc_loaders_from_starter(cfg: dict):
    """
    Robustly calls the RESISC45 loader from vlm.data. This tries common starter-code
    names: build_resisc45_loaders, build_resisc_loaders, build_resisc45_loader.
    """
    import vlm.data as data

    loader_fn = None
    for name in [
        "build_resisc45_loaders",
        "build_resisc_loaders",
        "build_resisc45_loader",
    ]:
        if hasattr(data, name):
            loader_fn = getattr(data, name)
            print(f"Using vlm.data.{name}")
            break

    if loader_fn is None:
        raise AttributeError(
            "Could not find a RESISC45 loader in vlm.data. "
            "Expected one of: build_resisc45_loaders, build_resisc_loaders, "
            "build_resisc45_loader."
        )

    sig = inspect.signature(loader_fn)

    possible_kwargs = {
        "img_size": cfg_get(cfg, "img_size", 64),
        "image_size": cfg_get(cfg, "img_size", 64),
        "batch_size": cfg_get(cfg, "batch_size", 128),
        "num_workers": cfg_get(cfg, "num_workers", 2),
        "seed": cfg_get(cfg, "seed", 0),
    }

    kwargs = {
        name: value
        for name, value in possible_kwargs.items()
        if name in sig.parameters
    }

    loaders = loader_fn(**kwargs)

    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    elif len(loaders) == 2:
        train_loader, test_loader = loaders
        val_loader = test_loader
    else:
        raise ValueError(f"Unexpected RESISC loader return format: {type(loaders)}")

    return train_loader, val_loader, test_loader


def unpack_batch(batch):
    """
    Supports:
      (images, labels)
      (images, captions, labels)
      {"image": ..., "label": ...}
    """
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        labels = batch.get("label", batch.get("labels"))
        return images, labels

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            images, labels = batch
            return images, labels
        if len(batch) >= 3:
            images = batch[0]
            labels = batch[-1]
            return images, labels

    raise ValueError(f"Unsupported batch format: {type(batch)}")


class ResiscClassifier(nn.Module):
    def __init__(self, vit: ViT, d_model: int, num_classes: int = 45) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_embeds = self.vit(images)
        logits = self.head(cls_embeds)
        return logits


def load_clip_pretrained_vit(
    checkpoint_path: Path,
    device: torch.device,
    img_size: int,
    patch_size: int,
    d_model: int,
    num_heads: int,
    num_blocks: int,
    dropout: float,
) -> ViT:
    vit = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "image_encoder" in ckpt:
        state_dict = ckpt["image_encoder"]
    elif "vit" in ckpt:
        state_dict = ckpt["vit"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = vit.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: missing ViT keys: {missing[:5]} ... total={len(missing)}")
    if unexpected:
        print(f"Warning: unexpected ViT keys: {unexpected[:5]} ... total={len(unexpected)}")

    return vit


def set_trainable_for_method(
    vit: ViT,
    method: str,
    rank: int,
    alpha: float,
) -> ViT:
    if method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad = False
        return vit

    if method == "lora":
        for p in vit.parameters():
            p.requires_grad = False
        vit = apply_lora_to_attention(vit, rank=rank, alpha=alpha)
        return vit

    if method == "full_ft":
        for p in vit.parameters():
            p.requires_grad = True
        return vit

    raise ValueError(f"Unknown method: {method}")


def count_parameters(model: nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total
    return total, trainable, ratio


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        images, labels = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    model.eval()

    total = 0
    correct = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        images, labels = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / total


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        if args.method == "lora":
            args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
        else:
            args.output_dir = Path("runs") / f"resisc_{args.method}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Method: {args.method}")

    img_size = cfg_get(cfg, "img_size", 64)
    patch_size = cfg_get(cfg, "patch_size", 8)
    d_model = cfg_get(cfg, "d_model", 384)
    num_heads = cfg_get(cfg, "num_heads", 6)
    num_blocks = cfg_get(cfg, "num_blocks", 6)
    dropout = cfg_get(cfg, "dropout", 0.1)

    num_epochs = cfg_get(cfg, "num_epochs", 10)
    lr = cfg_get(cfg, "lr", 3e-4)
    weight_decay = cfg_get(cfg, "weight_decay", 0.05)
    num_classes = cfg_get(cfg, "num_classes", 45)

    train_loader, val_loader, test_loader = build_resisc_loaders_from_starter(cfg)

    vit = load_clip_pretrained_vit(
        checkpoint_path=args.pretrained,
        device=device,
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    )

    vit = set_trainable_for_method(
        vit=vit,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
    )

    model = ResiscClassifier(
        vit=vit,
        d_model=d_model,
        num_classes=num_classes,
    ).to(device)

    # The classifier head should always be trainable.
    for p in model.head.parameters():
        p.requires_grad = True

    total_params, trainable_params, trainable_ratio = count_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_ratio:.6f}")
    print(f"Trainable percentage: {100 * trainable_ratio:.4f}%")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    train_losses = []
    val_accs = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_acc = evaluate_accuracy(
            model=model,
            loader=val_loader,
            device=device,
        )

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(f"train_loss = {train_loss:.4f}")
        print(f"val_acc = {val_acc:.4f}")

    test_acc = evaluate_accuracy(
        model=model,
        loader=test_loader,
        device=device,
    )

    wall_clock_seconds = time.perf_counter() - start_time

    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    else:
        peak_memory_bytes = 0
        peak_memory_mb = 0.0

    metrics = {
        "method": args.method,
        "rank": args.rank if args.method == "lora" else None,
        "alpha": args.alpha if args.method == "lora" else None,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "final_test_accuracy": test_acc,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_ratio,
        "peak_gpu_memory_bytes": peak_memory_bytes,
        "peak_gpu_memory_mb": peak_memory_mb,
        "wall_clock_seconds": wall_clock_seconds,
        "train_losses": train_losses,
        "val_accs": val_accs,
    }

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        args.output_dir / "last.pt",
    )

    print("\nDone.")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")
    print(f"Wall-clock training time: {wall_clock_seconds:.2f} seconds")
    print(f"Saved metrics to: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()