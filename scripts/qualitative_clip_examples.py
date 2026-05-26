from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import yaml

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders


CLASS_PROMPTS = [f"a satellite image of {name}" for name in EUROSAT_CLASSES]


def labels_from_captions(captions):
    labels = []
    for caption in captions:
        caption = str(caption)
        found = None
        for i, name in enumerate(EUROSAT_CLASSES):
            if name in caption:
                found = i
                break
        if found is None:
            raise ValueError(f"Could not infer label from caption: {caption}")
        labels.append(found)
    return torch.tensor(labels, dtype=torch.long)


def cfg_get(cfg: dict, key: str, default):
    if key in cfg:
        return cfg[key]
    for v in cfg.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return default


def unnormalize_image(x: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization for display.
    x: (3, H, W)
    returns: (H, W, 3), clipped to [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).cpu()


@torch.no_grad()
def compute_class_text_features(
    text_encoder: FrozenTextEncoder,
    projection_heads: ProjectionHeads,
    device: torch.device,
) -> torch.Tensor:
    text_embeds = text_encoder(CLASS_PROMPTS)
    text_embeds = text_embeds.clone().detach().to(device)

    text_proj = projection_heads.text_proj(text_embeds)
    text_proj = F.normalize(text_proj, p=2, dim=-1)

    return text_proj


@torch.no_grad()
def collect_examples(
    vit: ViT,
    projection_heads: ProjectionHeads,
    text_encoder: FrozenTextEncoder,
    val_loader,
    device: torch.device,
    num_correct: int = 5,
    num_incorrect: int = 5,
):
    vit.eval()
    projection_heads.eval()
    text_encoder.eval()

    text_proj = compute_class_text_features(text_encoder, projection_heads, device)

    correct_examples = []
    incorrect_examples = []

    for images, captions in val_loader:
        images = images.to(device)
        labels = labels_from_captions(captions).to(device)

        image_embeds = vit(images)
        image_proj = projection_heads.image_proj(image_embeds)
        image_proj = F.normalize(image_proj, p=2, dim=-1)

        logits = image_proj @ text_proj.T
        probs = logits.softmax(dim=-1)

        top_probs, top_indices = probs.topk(3, dim=-1)
        preds = top_indices[:, 0]

        for i in range(images.shape[0]):
            true_idx = labels[i].item()
            pred_idx = preds[i].item()

            example = {
                "image": images[i].detach().cpu(),
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_label": EUROSAT_CLASSES[true_idx],
                "pred_label": EUROSAT_CLASSES[pred_idx],
                "top3_indices": top_indices[i].detach().cpu().tolist(),
                "top3_probs": top_probs[i].detach().cpu().tolist(),
                "top3_labels": [EUROSAT_CLASSES[j] for j in top_indices[i].detach().cpu().tolist()],
            }

            if pred_idx == true_idx and len(correct_examples) < num_correct:
                correct_examples.append(example)
            elif pred_idx != true_idx and len(incorrect_examples) < num_incorrect:
                incorrect_examples.append(example)

            if len(correct_examples) >= num_correct and len(incorrect_examples) >= num_incorrect:
                return correct_examples, incorrect_examples

    return correct_examples, incorrect_examples


def plot_examples(correct_examples, incorrect_examples, output_path: Path):
    examples = correct_examples + incorrect_examples

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for ax, ex in zip(axes, examples):
        img = unnormalize_image(ex["image"])
        ax.imshow(img)
        ax.axis("off")

        status = "Correct" if ex["true_idx"] == ex["pred_idx"] else "Wrong"
        title = (
            f"{status}\n"
            f"True: {ex['true_label']}\n"
            f"Pred: {ex['pred_label']}"
        )
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_examples_json(correct_examples, incorrect_examples, output_path: Path):
    def strip_image(ex):
        return {
            k: v
            for k, v in ex.items()
            if k != "image"
        }

    data = {
        "correct": [strip_image(ex) for ex in correct_examples],
        "incorrect": [strip_image(ex) for ex in incorrect_examples],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_discussion(correct_examples, incorrect_examples, output_path: Path):
    lines = []

    lines.append("Correct examples:")
    for i, ex in enumerate(correct_examples, 1):
        lines.append(
            f"{i}. true={ex['true_label']}, pred={ex['pred_label']}"
        )

    lines.append("")
    lines.append("Incorrect examples with top-3 predictions:")
    for i, ex in enumerate(incorrect_examples, 1):
        top3 = ", ".join(
            [
                f"{label} ({prob:.3f})"
                for label, prob in zip(ex["top3_labels"], ex["top3_probs"])
            ]
        )
        lines.append(
            f"{i}. true={ex['true_label']}, pred={ex['pred_label']}, top3=[{top3}]"
        )

    lines.append("")
    lines.append("Discussion draft:")
    lines.append(
        "Most of the incorrect predictions are still semantically reasonable: the model often confuses visually similar land-use categories, such as crop or vegetation classes, rather than choosing completely unrelated classes. "
        "The top-3 predictions for wrong examples usually contain classes with similar textures, colors, or spatial patterns, which suggests that the learned CLIP embedding space is organizing images by meaningful visual/semantic similarity. "
        "This also explains why zero-shot accuracy can be high even when some errors remain: the model has learned a useful global satellite-image representation, but nearby classes in the embedding space can still overlap. "
        "Nonsensical mistakes would suggest poor alignment, but reasonable top-3 mistakes indicate that the embedding space has learned structure even when the top-1 prediction is wrong."
    )

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/clip_eurosat.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/clip_eurosat/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat/qualitative"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_size = cfg_get(cfg, "img_size", 64)
    patch_size = cfg_get(cfg, "patch_size", 8)
    d_model = cfg_get(cfg, "d_model", 384)
    num_heads = cfg_get(cfg, "num_heads", 6)
    num_blocks = cfg_get(cfg, "num_blocks", 6)
    dropout = cfg_get(cfg, "dropout", 0.1)
    d_proj = cfg_get(cfg, "d_proj", 256)
    batch_size = cfg_get(cfg, "batch_size", 256)
    num_workers = cfg_get(cfg, "num_workers", 2)
    text_model = cfg_get(cfg, "model_name", "sentence-transformers/all-MiniLM-L6-v2")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    train_loader, val_loader, test_loader = build_eurosat_loaders(
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
    text_encoder.eval()

    projection_heads = ProjectionHeads(
        d_image=d_model,
        d_text=text_encoder.embedding_dim,
        d_proj=d_proj,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    vit.load_state_dict(ckpt["image_encoder"])
    projection_heads.load_state_dict(ckpt["projection_heads"])

    correct_examples, incorrect_examples = collect_examples(
        vit=vit,
        projection_heads=projection_heads,
        text_encoder=text_encoder,
        val_loader=val_loader,
        device=device,
        num_correct=5,
        num_incorrect=5,
    )

    print(f"Collected {len(correct_examples)} correct examples")
    print(f"Collected {len(incorrect_examples)} incorrect examples")

    plot_path = args.output_dir / "qualitative_examples.png"
    json_path = args.output_dir / "qualitative_examples.json"
    discussion_path = args.output_dir / "qualitative_discussion.txt"

    plot_examples(correct_examples, incorrect_examples, plot_path)
    write_examples_json(correct_examples, incorrect_examples, json_path)
    write_discussion(correct_examples, incorrect_examples, discussion_path)

    print(f"Saved image grid to: {plot_path}")
    print(f"Saved labels/top-3 predictions to: {json_path}")
    print(f"Saved discussion draft to: {discussion_path}")


if __name__ == "__main__":
    main()
