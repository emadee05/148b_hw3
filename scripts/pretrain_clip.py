from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from basics.vit import ViT
from basics.text_encoder import FrozenTextEncoder
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from vlm.data import build_eurosat_loaders


EUROSAT_CLASSES = [
    "annual crop land",
    "forest",
    "herbaceous vegetation land",
    "highway or road",
    "industrial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings",
    "river",
    "sea or lake",
]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cfg_get(cfg: dict, key: str, default):
    """Allows either flat YAML keys or nested config dicts."""
    if key in cfg:
        return cfg[key]
    for v in cfg.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return default


def make_loaders(cfg: dict):
    """
    Calls build_eurosat_loaders using whatever argument names your starter
    implementation supports.
    """
    sig = inspect.signature(build_eurosat_loaders)
    kwargs = {}

    possible = {
        "img_size": cfg_get(cfg, "img_size", 64),
        "image_size": cfg_get(cfg, "img_size", 64),
        "batch_size": cfg_get(cfg, "batch_size", 256),
        "num_workers": cfg_get(cfg, "num_workers", 2),
        "seed": cfg_get(cfg, "seed", 0),
    }

    for name in sig.parameters:
        if name in possible:
            kwargs[name] = possible[name]

    return build_eurosat_loaders(**kwargs)


def unpack_batch(batch):
    """
    Supports common batch formats:
      (images, captions)
      (images, captions, labels)
      {"image": ..., "caption": ..., "label": ...}
    """
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        captions = batch.get("caption", batch.get("captions"))
        labels = batch.get("label", batch.get("labels", None))
        return images, captions, labels

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            images, captions = batch
            labels = None
            return images, captions, labels
        if len(batch) >= 3:
            images, captions, labels = batch[:3]
            return images, captions, labels

    raise ValueError(f"Unsupported batch format: {type(batch)}")


def encode_text(text_encoder, captions, device):
    with torch.no_grad():
        text_embeds = text_encoder(list(captions))
    return text_embeds.to(device)


@torch.no_grad()
def zero_shot_accuracy(image_encoder, projection_heads, text_encoder, val_loader, device):
    image_encoder.eval()
    projection_heads.eval()
    text_encoder.eval()

    class_prompts = [f"a satellite image of {name}" for name in EUROSAT_CLASSES]

    class_text_embeds = text_encoder(class_prompts).to(device)

    # Make dummy image embeddings only to use the text projection head cleanly.
    # We only need projected text vectors here.
    text_proj = projection_heads.text_proj(class_text_embeds)
    text_proj = nn.functional.normalize(text_proj, p=2, dim=-1)

    total = 0
    correct = 0

    for batch in val_loader:
        images, captions, labels = unpack_batch(batch)

        if labels is None:
            continue

        images = images.to(device)
        labels = labels.to(device)

        image_embeds = image_encoder(images)
        image_proj = projection_heads.image_proj(image_embeds)
        image_proj = nn.functional.normalize(image_proj, p=2, dim=-1)

        logits = image_proj @ text_proj.T
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    if total == 0:
        return float("nan")

    return correct / total


def train_one_epoch(
    image_encoder,
    projection_heads,
    text_encoder,
    logit_scale,
    train_loader,
    optimizer,
    device,
):
    image_encoder.train()
    projection_heads.train()
    text_encoder.eval()

    losses = []

    pbar = tqdm(train_loader, desc="train", leave=False)

    for batch in pbar:
        images, captions, labels = unpack_batch(batch)
        images = images.to(device)

        image_embeds = image_encoder(images)
        text_embeds = encode_text(text_encoder, captions, device)

        image_proj, text_proj = projection_heads(image_embeds, text_embeds)

        loss = clip_loss(image_proj, text_proj, logit_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # CLIP-style clamp to prevent runaway inverse temperature growth.
        with torch.no_grad():
            logit_scale.clamp_(max=math.log(100.0))

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", scale=f"{logit_scale.exp().item():.2f}")

    return sum(losses) / len(losses)


def plot_curve(values, ylabel, title, save_path):
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clip_eurosat.yaml")
    parser.add_argument("--out_dir", type=str, default="runs/clip_eurosat")
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    num_epochs = cfg_get(cfg, "num_epochs", 20)

    train_loader, val_loader, test_loader = make_loaders(cfg)

    image_encoder = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    text_encoder = FrozenTextEncoder("sentence-transformers/all-MiniLM-L6-v2")
    text_encoder.eval()

    d_text = text_encoder.embedding_dim

    projection_heads = ProjectionHeads(
        d_image=d_model,
        d_text=d_text,
        d_proj=d_proj,
    ).to(device)

    logit_scale = init_logit_scale().to(device)

    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters())
        + list(projection_heads.parameters())
        + [logit_scale],
        lr=lr,
        weight_decay=weight_decay,
    )

    train_losses = []
    val_accs = []

    best_val_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            image_encoder=image_encoder,
            projection_heads=projection_heads,
            text_encoder=text_encoder,
            logit_scale=logit_scale,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_acc = zero_shot_accuracy(
            image_encoder=image_encoder,
            projection_heads=projection_heads,
            text_encoder=text_encoder,
            val_loader=val_loader,
            device=device,
        )

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(f"train_loss = {train_loss:.4f}")
        print(f"zero_shot_val_acc = {val_acc:.4f}")
        print(f"logit_scale.exp() = {logit_scale.exp().item():.4f}")

        checkpoint = {
            "epoch": epoch,
            "image_encoder": image_encoder.state_dict(),
            "projection_heads": projection_heads.state_dict(),
            "logit_scale": logit_scale.detach().cpu(),
            "train_losses": train_losses,
            "val_accs": val_accs,
            "config": cfg,
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if not math.isnan(val_acc) and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, out_dir / "best.pt")

        plot_curve(
            train_losses,
            ylabel="Training loss",
            title="CLIP pretraining loss on EuroSAT",
            save_path=out_dir / "train_loss_curve.png",
        )

        plot_curve(
            val_accs,
            ylabel="Zero-shot validation accuracy",
            title="Zero-shot validation accuracy on EuroSAT",
            save_path=out_dir / "val_accuracy_curve.png",
        )

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "val_accs": val_accs,
                    "best_val_acc": best_val_acc,
                },
                f,
                indent=2,
            )

    print("\nDone.")
    print(f"Saved training-loss curve to: {out_dir / 'train_loss_curve.png'}")
    print(f"Saved validation-accuracy curve to: {out_dir / 'val_accuracy_curve.png'}")
    print(f"Saved best checkpoint to: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
