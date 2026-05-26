"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
import json
import time
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.vit import ViT
from vlm.projector import VisionLanguageProjector
from vlm.model import VisionLanguageModel
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy
from basics.lora import LoRALinear

@torch.no_grad()
def evaluate_clevr_accuracy(
    model,
    val_loader,
    device,
    injection: str,
    max_examples: int = 500,
) -> dict[str, float]:
    model.eval()

    predictions = []
    golds = []
    q_types = []

    for batch in val_loader:
        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        batch_q_types = batch.get("q_type", [None] * len(questions))

        if injection == "interleaved":
            prompts = [f"Question: <image> {q} Answer:" for q in questions]
        else:
            prompts = [f"Question: {q} Answer:" for q in questions]

        outputs = model.generate(
            images=images,
            prompts=prompts,
            injection=injection,
            max_new_tokens=8,
            do_sample=False,
        )

        # Basic cleanup: keep only the answer-looking suffix.
        for out in outputs:
            if "Answer:" in out:
                pred = out.split("Answer:")[-1].strip()
            else:
                pred = out.strip()

            # Often generated answers include extra text; take first token/phrase line.
            pred = pred.split("\n")[0].strip()
            predictions.append(pred)

        golds.extend([str(a).strip() for a in answers])
        q_types.extend(batch_q_types)

        if len(predictions) >= max_examples:
            predictions = predictions[:max_examples]
            golds = golds[:max_examples]
            q_types = q_types[:max_examples]
            break

    from vlm.eval import batch_clevr_accuracy

    metrics = batch_clevr_accuracy(
        predictions=predictions,
        golds=golds,
        q_types=q_types,
    )

    model.train()
    return metrics

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in.
    # Sketch:
    #   1. Build CLEVR loaders via vlm.data.build_clevr_loaders.
    #   2. Load CLIP-pretrained ViT (args.pretrained_vit).
    #   3. Load SmolLM2-360M-Instruct decoder + tokenizer in bf16 with FA2.
    #      - Add the special <image> token to the tokenizer if injection ==
    #        "interleaved", and resize_token_embeddings on the decoder.
    #   4. Build VisionLanguageProjector and VisionLanguageModel.
    #   5. Apply the chosen freeze configuration:
    #        A: vit frozen, projector trained, decoder frozen.
    #        B: vit frozen, projector trained, decoder LoRA.
    #        C: vit frozen, projector trained, decoder full FT.
    #        D: everything full FT.
    #   6. Train for cfg["num_steps"] with bf16 gradient accumulation.
    #   7. Periodically run vlm.eval.batch_clevr_accuracy on the val set,
    #      log: val accuracy, peak memory, train loss, gradient norm.
    #   8. Save best checkpoint.
    device = torch.device(args.device)

    # -----------------------------
    # 1. Config values
    # -----------------------------
    img_size = cfg.get("img_size", 64)
    patch_size = cfg.get("patch_size", 8)
    d_model = cfg.get("d_model", 384)
    num_heads = cfg.get("num_heads", 6)
    num_blocks = cfg.get("num_blocks", 6)
    dropout = cfg.get("dropout", 0.1)

    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-4)
    num_steps = cfg.get("num_steps", 2000)
    val_every = cfg.get("val_every", 200)
    num_workers = cfg.get("num_workers", 2)

    decoder_name = cfg.get("decoder_name", "HuggingFaceTB/SmolLM2-360M-Instruct")

    # -----------------------------
    # 2. CLEVR loaders
    # -----------------------------
    train_loader, val_loader = build_clevr_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_iter = iter(train_loader)

    # -----------------------------
    # 3. Load CLIP-pretrained ViT
    # -----------------------------
    vit = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    ckpt = torch.load(args.pretrained_vit, map_location=device)
    vit.load_state_dict(ckpt["image_encoder"], strict=False)

    # -----------------------------
    # 4. Load decoder/tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    if args.injection == "interleaved":
        decoder.resize_token_embeddings(len(tokenizer))

    d_decoder = decoder.config.hidden_size

    # -----------------------------
    # 5. Projector + VLM
    # -----------------------------
    projector = VisionLanguageProjector(
        d_image=d_model,
        d_decoder=d_decoder,
        expansion=4,
    ).to(device)

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    ).to(device)

    # Always freeze ViT unless config D
    for p in model.vit.parameters():
        p.requires_grad = args.freeze_config == "D"

    # Always train projector
    for p in model.projector.parameters():
        p.requires_grad = True

    # Decoder depends on config
    if args.freeze_config == "A":
        for p in model.decoder.parameters():
            p.requires_grad = False

    elif args.freeze_config == "B":
        for p in model.decoder.parameters():
            p.requires_grad = False

        # Wrap q_proj and v_proj in every attention layer with LoRA
        for module in model.decoder.modules():
            if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                module.q_proj = LoRALinear(module.q_proj, rank=8, alpha=16)
            if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
                module.v_proj = LoRALinear(module.v_proj, rank=8, alpha=16)

    elif args.freeze_config in {"C", "D"}:
        for p in model.decoder.parameters():
            p.requires_grad = True

    # IMPORTANT: move again after applying LoRA.
    # LoRA creates new parameters, and new PyTorch params start on CPU by default.
    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params:.6f}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    # -----------------------------
    # 7. Training loop
    # -----------------------------
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    losses = []
    val_accs = []
    best_val_acc = -1.0

    model.train()

    for step in range(1, num_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

        clean_questions = [str(q).replace("<image>", "").strip() for q in questions]

        if args.injection == "interleaved":
            prompts = [f"Question: <image> {q} Answer:" for q in clean_questions]
        else:
            prompts = [f"Question: {q} Answer:" for q in clean_questions]

        full_text = [p + " " + str(a) for p, a in zip(prompts, answers)]
        
        tokenized = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            injection=args.injection,
            mask_mode=args.mask_mode,
        )

        loss = outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0:
            print(f"step {step}/{num_steps}, loss={loss.item():.4f}")

        if step % val_every == 0 or step == num_steps:
            val_metrics = evaluate_clevr_accuracy(
                model=model,
                val_loader=val_loader,
                device=device,
                injection=args.injection,
                max_examples=500,
            )

            val_acc = val_metrics.get("overall", val_metrics.get("accuracy", 0.0))
            val_accs.append({"step": step, "val_exact_match": val_acc})
            print(f"VAL step {step}: exact_match={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "projector": model.projector.state_dict(),
                        "config": cfg,
                        "args": vars(args),
                        "step": step,
                        "val_exact_match": val_acc,
                    },
                    args.output_dir / "best.pt",
                )

    wall_clock = time.perf_counter() - start_time
    wall_clock_per_step = wall_clock / num_steps

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    num_visual_tokens = {
        "cls": 1,
        "all_patches": (img_size // patch_size) ** 2 + 1,
        "interleaved": (img_size // patch_size) ** 2 + 1,
    }[args.injection]

    metrics = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "lr": lr,
        "num_visual_tokens": num_visual_tokens,
        "best_val_exact_match": best_val_acc,
        "peak_gpu_memory_mb": peak_memory_mb,
        "wall_clock_seconds": wall_clock,
        "wall_clock_per_step": wall_clock_per_step,
        "trainable_params": trainable_params,
        "losses": losses,
        "val_accs": val_accs,
    }

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(f"Best val exact match: {best_val_acc:.4f}")
    print(f"Visual tokens/example: {num_visual_tokens}")
    print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")
    print(f"Wall-clock time/step: {wall_clock_per_step:.4f} s")
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
