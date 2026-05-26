"""§3 — CLIP-style pretraining on EuroSAT.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
from vlm.eval import zeroshot_classification_accuracy

CLASS_PROMPTS = [f"a satellite image of {name}" for name in EUROSAT_CLASSES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def cfg_get(cfg: dict, key: str, default):
    if key in cfg:
        return cfg[key]
    for v in cfg.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return default


def train_one_epoch(
    vit: ViT,
    projection_heads: ProjectionHeads,
    text_encoder: FrozenTextEncoder,
    logit_scale: torch.Tensor,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    vit.train()
    projection_heads.train()
    text_encoder.eval()

    losses: list[float] = []
    for images, captions in tqdm(train_loader, desc="train", leave=False):
        images = images.to(device)

        image_embeds = vit(images)
        with torch.no_grad():
            text_embeds = text_encoder(list(captions))

        # FrozenTextEncoder may return an inference tensor; clone/detach makes it
        # safe to pass through the trainable projection head during backprop.
        text_embeds = text_embeds.clone().detach().to(device)

        image_proj, text_proj = projection_heads(image_embeds, text_embeds)
        loss = clip_loss(image_proj, text_proj, logit_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            logit_scale.clamp_(max=math.log(100.0))

        losses.append(loss.item())

    return sum(losses) / len(losses)


def plot_curve(values: list[float], ylabel: str, title: str, path: Path) -> None:
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_curves_analysis(
    train_losses: list[float], val_accs: list[float], path: Path
) -> None:
    if len(train_losses) < 2:
        path.write_text("Run full training to compare curves.\n")
        return

    tail = max(1, len(train_losses) // 4)
    late_loss_drop = train_losses[-tail] - train_losses[-1]
    late_acc_gain = val_accs[-1] - val_accs[-tail]
    best_acc = max(val_accs)

    text = (
        f"Training loss fell from {train_losses[0]:.3f} to {train_losses[-1]:.3f} "
        f"while zero-shot validation accuracy rose from {val_accs[0]:.3f} to "
        f"{val_accs[-1]:.3f} (best {best_acc:.3f}). "
    )
    if late_acc_gain < 0.01 and late_loss_drop > 0.02:
        text += (
            "In the last quarter of training, validation accuracy plateaued "
            f"({late_acc_gain:+.3f}) but loss still decreased ({late_loss_drop:.3f}), "
            "so yes—training loss can keep improving after accuracy plateaus. "
        )
    else:
        text += (
            "Both curves usually improve together early on; check the PNG plots "
            "for epoch-by-epoch divergence. "
        )
    text += (
        "EuroSAT uses repeated class-template captions within batches, so InfoNCE "
        "loss can shrink from easy duplicate positives without much extra "
        "zero-shot gain."
    )
    path.write_text(text + "\n")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    img_size = cfg_get(cfg, "img_size", 64)
    patch_size = cfg_get(cfg, "patch_size", 8)
    d_model = cfg_get(cfg, "d_model", 384)
    num_heads = cfg_get(cfg, "num_heads", 6)
    num_blocks = cfg_get(cfg, "num_blocks", 6)
    dropout = cfg_get(cfg, "dropout", 0.1)
    d_proj = cfg_get(cfg, "d_proj", 256)
    lr = cfg_get(cfg, "lr", 3e-4)
    weight_decay = cfg_get(cfg, "weight_decay", 0.1)
    betas = tuple(cfg_get(cfg, "betas", [0.9, 0.95]))
    num_epochs = cfg_get(cfg, "num_epochs", 20)
    batch_size = cfg_get(cfg, "batch_size", 256)
    num_workers = cfg_get(cfg, "num_workers", 4)
    warmup_steps = cfg_get(cfg, "warmup_steps", 0)
    text_model = cfg_get(cfg, "model_name", "sentence-transformers/all-MiniLM-L6-v2")

    train_loader, val_loader, _test_loader = build_eurosat_loaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    vit = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    text_encoder = FrozenTextEncoder(text_model)
    projection_heads = ProjectionHeads(
        d_image=d_model,
        d_text=text_encoder.embedding_dim,
        d_proj=d_proj,
    ).to(device)

    logit_scale = init_logit_scale()
    logit_scale.data = logit_scale.data.to(device)
    optimizer = torch.optim.AdamW(
        list(vit.parameters())
        + list(projection_heads.parameters())
        + [logit_scale],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = None
    if cfg_get(cfg, "scheduler", None) == "cosine" and total_steps > warmup_steps:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project="hw3-clip", config=cfg)

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(
            vit,
            projection_heads,
            text_encoder,
            logit_scale,
            train_loader,
            optimizer,
            scheduler,
            device,
        )
        val_acc = zeroshot_classification_accuracy(
            vit=vit,
            projection_heads=projection_heads,
            text_encoder=text_encoder,
            val_loader=val_loader,
            class_prompts=CLASS_PROMPTS,
            class_indices=list(range(len(CLASS_PROMPTS))),
            device=device,
        )

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(f"train_loss = {train_loss:.4f}")
        print(f"zero_shot_val_acc = {val_acc:.4f}")
        print(f"logit_scale.exp() = {logit_scale.exp().item():.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train_loss": train_loss,
                    "val_zero_shot_acc": val_acc,
                    "logit_scale": logit_scale.exp().item(),
                    "epoch": epoch,
                }
            )

        checkpoint = {
            "epoch": epoch,
            "image_encoder": vit.state_dict(),
            "projection_heads": projection_heads.state_dict(),
            "logit_scale": logit_scale.detach().cpu(),
            "train_losses": train_losses,
            "val_accs": val_accs,
            "config": cfg,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, args.output_dir / "best.pt")

        plot_curve(
            train_losses,
            "Training loss",
            "CLIP pretraining loss on EuroSAT",
            args.output_dir / "train_loss_curve.png",
        )
        plot_curve(
            val_accs,
            "Zero-shot validation accuracy",
            "Zero-shot validation accuracy on EuroSAT",
            args.output_dir / "val_accuracy_curve.png",
        )
        with open(args.output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "val_accs": val_accs,
                    "best_val_acc": best_val_acc,
                },
                f,
                indent=2,
            )

    write_curves_analysis(
        train_losses, val_accs, args.output_dir / "curves_analysis.txt"
    )

    print("\nDone.")
    print(f"(a) {args.output_dir / 'train_loss_curve.png'}")
    print(f"(b) {args.output_dir / 'val_accuracy_curve.png'}")
    print(f"(c) {args.output_dir / 'curves_analysis.txt'}")


if __name__ == "__main__":
    main()
