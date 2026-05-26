"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.vit import ViT
from basics.lora import LoRALinear
from vlm.projector import VisionLanguageProjector
from vlm.model import VisionLanguageModel
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def normalize_answer(s: str) -> str:
    return str(s).lower().strip().replace(".", "").replace(",", "")


def clean_generation(out: str) -> str:
    if "Answer:" in out:
        out = out.split("Answer:")[-1]

    out = out.strip()

    # Stop if the model starts generating another prompt/question.
    for stop in ["Question:", "Q:", "\n"]:
        if stop in out:
            out = out.split(stop)[0].strip()

    # Remove punctuation at the end.
    out = out.strip(" .,\n\t")

    return out

def apply_lora_to_decoder_qv(decoder: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    """Wrap SmolLM/LLaMA q_proj and v_proj layers with LoRA."""
    for module in decoder.modules():
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            module.q_proj = LoRALinear(module.q_proj, rank=rank, alpha=alpha)
        if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
            module.v_proj = LoRALinear(module.v_proj, rank=rank, alpha=alpha)


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> VisionLanguageModel:
    cfg = ckpt["config"]
    ckpt_args = ckpt.get("args", {})

    img_size = cfg.get("img_size", 64)
    patch_size = cfg.get("patch_size", 8)
    d_model = cfg.get("d_model", 384)
    num_heads = cfg.get("num_heads", 6)
    num_blocks = cfg.get("num_blocks", 6)
    dropout = cfg.get("dropout", 0.1)
    decoder_name = cfg.get("decoder_name", "HuggingFaceTB/SmolLM2-360M-Instruct")

    injection = ckpt_args.get("injection", "all_patches")
    freeze_config = ckpt_args.get("freeze_config", "A")

    vit = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = None
    if injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    if injection == "interleaved":
        decoder.resize_token_embeddings(len(tokenizer))

    # Important for config B checkpoints: recreate LoRA modules before loading state_dict.
    if freeze_config == "B":
        apply_lora_to_decoder_qv(decoder, rank=8, alpha=16.0)
        decoder = decoder.to(device)

    projector = VisionLanguageProjector(
        d_image=d_model,
        d_decoder=decoder.config.hidden_size,
        expansion=4,
    ).to(device)

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def save_image_tensor(img: torch.Tensor, path: Path) -> None:
    """Save CHW image tensor as PNG."""
    img = img.detach().cpu().float()
    img = img.permute(1, 2, 0)

    # Normalize for display.
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    plt.imsave(path, img.numpy())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_images:
        image_dir = args.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
    else:
        image_dir = None

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    ckpt_args = ckpt.get("args", {})

    injection = ckpt_args.get("injection", "all_patches")

    model = build_model_from_checkpoint(ckpt, device)

    _, val_loader = build_clevr_loaders(
        batch_size=1,
        num_workers=cfg.get("num_workers", 2),
    )

    predictions = []
    golds = []
    q_types = []

    qualitative_rows = []
    correct_rows = []
    incorrect_rows = []

    total_seen = 0

    for batch in val_loader:
        if total_seen >= args.max_eval:
            break

        image = batch["image"].to(device)
        question = str(batch["question"][0])
        gold = str(batch["answer"][0])
        q_type = batch.get("q_type", ["unknown"])[0]

        clean_question = question.replace("<image>", "").strip()

        if injection == "interleaved":
            prompt = f"Question: <image> {clean_question} Answer:"
        else:
            prompt = f"Question: {clean_question} Answer:"

        outputs = model.generate(
            images=image,
            prompts=[prompt],
            injection=injection,
            max_new_tokens=8,
            do_sample=False,
        )

        pred = clean_generation(outputs[0])
        correct = normalize_answer(pred) == normalize_answer(gold)

        predictions.append(pred)
        golds.append(gold)
        q_types.append(q_type)

        row = {
            "index": total_seen,
            "image_file": None,
            "question": question,
            "q_type": q_type,
            "gold": gold,
            "prediction": pred,
            "correct": correct,
        }

        if args.save_images:
            image_file = f"example_{total_seen:04d}.png"
            save_image_tensor(image[0], image_dir / image_file)
            row["image_file"] = str(Path("images") / image_file)

        if correct:
            correct_rows.append(row)
        else:
            incorrect_rows.append(row)

        total_seen += 1

    metrics = batch_clevr_accuracy(
        predictions=predictions,
        golds=golds,
        q_types=q_types,
    )

    # Pick a mix of correct and incorrect examples.
    qualitative_rows = correct_rows[: args.num_examples // 2] + incorrect_rows[: args.num_examples]
    qualitative_rows = qualitative_rows[: args.num_examples]

    # If not enough correct examples, fill with incorrect ones.
    if len(qualitative_rows) < args.num_examples:
        all_rows = correct_rows + incorrect_rows
        qualitative_rows = all_rows[: args.num_examples]

    jsonl_path = args.output_dir / "examples.jsonl"
    with open(jsonl_path, "w") as f:
        for row in qualitative_rows:
            f.write(json.dumps(row) + "\n")

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAccuracy metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved qualitative examples to {jsonl_path}")
    print(f"Saved metrics to {metrics_path}")

    print("\nQualitative examples:")
    for row in qualitative_rows:
        status = "CORRECT" if row["correct"] else "WRONG"
        print(f"\n[{status}]")
        print(f"image: {row['image_file']}")
        print(f"q_type: {row['q_type']}")
        print(f"Q: {row['question']}")
        print(f"GT: {row['gold']}")
        print(f"Pred: {row['prediction']}")


if __name__ == "__main__":
    main()

