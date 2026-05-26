"""Sweep LoRA rank on RESISC45.

This script calls scripts/finetune_resisc.py multiple times, once for each LoRA
rank r in {1, 2, 4, 8, 16, 32, 64}, using alpha = 2r.

Usage:
    uv run python scripts/sweep_lora_rank.py \
        --config configs/lora_resisc.yaml \
        --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANKS = [1, 2, 4, 8, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lora_resisc.yaml"),
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("runs/clip_eurosat/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/resisc_lora_sweep"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing metrics.json files instead of rerunning completed ranks.",
    )
    return parser.parse_args()


def run_one_lora_rank(
    rank: int,
    config_path: Path,
    pretrained_path: Path,
    output_root: Path,
    skip_existing: bool = False,
) -> dict:
    alpha = 2 * rank
    rank_dir = output_root / f"rank{rank}"
    metrics_path = rank_dir / "metrics.json"

    if skip_existing and metrics_path.exists():
        print(f"Skipping rank {rank}; found {metrics_path}")
    else:
        rank_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/finetune_resisc.py",
            "--config",
            str(config_path),
            "--method",
            "lora",
            "--rank",
            str(rank),
            "--alpha",
            str(alpha),
            "--pretrained",
            str(pretrained_path),
            "--output-dir",
            str(rank_dir),
        ]

        print("\n" + "=" * 100)
        print(f"Running LoRA rank r={rank}, alpha={alpha}")
        print(" ".join(cmd))
        print("=" * 100)

        subprocess.run(cmd, check=True)

    with open(metrics_path) as f:
        metrics = json.load(f)

    return metrics


def run_lora_rank_sweep(
    config_path: Path,
    pretrained_path: Path,
    output_root: Path,
    skip_existing: bool = False,
) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)

    results = []

    for rank in RANKS:
        metrics = run_one_lora_rank(
            rank=rank,
            config_path=config_path,
            pretrained_path=pretrained_path,
            output_root=output_root,
            skip_existing=skip_existing,
        )
        results.append(metrics)

    results = sorted(results, key=lambda m: m["rank"])

    with open(output_root / "all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_lora_rank_sweep(results: list[dict], output_path: Path) -> None:
    ranks = [m["rank"] for m in results]
    test_accs = [m["final_test_accuracy"] for m in results]

    plt.figure(figsize=(7, 5))
    plt.plot(ranks, test_accs, marker="o")
    plt.xscale("log", base=2)
    plt.xticks(ranks, [str(r) for r in ranks])
    plt.xlabel("LoRA rank $r$")
    plt.ylabel("RESISC45 test accuracy")
    plt.title("LoRA rank sweep on RESISC45")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_lora_rank_table(results: list[dict], output_path: Path) -> None:
    lines = []
    lines.append(
        "| Rank r | Alpha | Test accuracy | Trainable params | "
        "Peak GPU memory (MB) | Time (s) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")

    for m in results:
        lines.append(
            f"| {m['rank']} | {m['alpha']} | "
            f"{m['final_test_accuracy']:.4f} | "
            f"{m['trainable_parameters']:,} | "
            f"{m['peak_gpu_memory_mb']:.2f} | "
            f"{m['wall_clock_seconds']:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def write_discussion_template(results: list[dict], output_path: Path) -> None:
    best = max(results, key=lambda m: m["final_test_accuracy"])
    lines = []

    lines.append("Discussion template:")
    lines.append(
        f"The best rank in this sweep was r={best['rank']} with test accuracy "
        f"{best['final_test_accuracy']:.4f}. From the plot, diminishing returns "
        "appear once increasing the rank gives only small additional gains in "
        "accuracy despite increasing the number of trainable parameters. "
        "This supports the LoRA intuition that the useful fine-tuning update is "
        "effectively low-rank: after some moderate rank, extra adapter capacity "
        "does not substantially change the downstream classifier. Compared with "
        "common practical LoRA ranks such as r=8 or r=16 in large-model fine-tuning, "
        "the observed saturation point tells us whether this smaller ViT/RESISC45 "
        "setting needs similarly low-rank updates or benefits from a larger rank."
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    results = run_lora_rank_sweep(
        config_path=args.config,
        pretrained_path=args.pretrained,
        output_root=args.output_dir,
        skip_existing=args.skip_existing,
    )

    plot_path = args.output_dir / "lora_rank_sweep.png"
    table_path = args.output_dir / "lora_rank_sweep_table.md"
    discussion_path = args.output_dir / "lora_rank_discussion_template.txt"

    plot_lora_rank_sweep(results, plot_path)
    write_lora_rank_table(results, table_path)
    write_discussion_template(results, discussion_path)

    print("\nDone.")
    print(f"Saved all metrics to: {args.output_dir / 'all_metrics.json'}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved table to: {table_path}")
    print(f"Saved discussion template to: {discussion_path}")

    print("\nSummary table:")
    print(table_path.read_text())


if __name__ == "__main__":
    main()
