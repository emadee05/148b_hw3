"""Benchmark ViT forward pass wall time for several patch sizes P.

Usage:
  uv run python scripts/benchmark_vit_forward.py
  uv run python scripts/benchmark_vit_forward.py --img-size 224 --patch-sizes 8,16,32
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from basics.vit import ViT


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT forward timing (CUDA).")
    parser.add_argument("--img-size", type=int, default=224, help="Square image side length.")
    parser.add_argument(
        "--patch-sizes",
        type=str,
        default="8,16,32",
        help="Comma-separated patch sizes P (must divide img-size).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for torch.cuda.synchronize() timing.")

    device = torch.device("cuda")
    patch_sizes = [int(p.strip()) for p in args.patch_sizes.split(",") if p.strip()]

    for p in patch_sizes:
        if args.img_size % p != 0:
            raise SystemExit(f"img_size {args.img_size} must be divisible by P={p}")

        model = ViT(
            img_size=args.img_size,
            patch_size=p,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
        ).to(device)
        model.eval()

        x = torch.randn(
            args.batch_size, 3, args.img_size, args.img_size, device=device
        )

        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model(x)
            torch.cuda.synchronize()

            times_ms: list[float] = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(times_ms)
        stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        n_patches = (args.img_size // p) ** 2
        print(
            f"P={p:3d}  seq_len={n_patches + 1:4d}  "
            f"mean={mean_ms:.3f} ms  stdev={stdev_ms:.3f} ms  (batch={args.batch_size})"
        )


if __name__ == "__main__":
    main()
