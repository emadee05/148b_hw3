import time
import torch
import pandas as pd

from basics.vit import ViT


def make_vit(patch_size):
    return ViT(
        img_size=224,
        patch_size=patch_size,
        d_model=384,
        num_heads=6,
        num_blocks=6,
        dropout=0.1,
    )


def benchmark_patch_size(patch_size, device):
    batch_size = 16
    image_size = 224
    warmup_steps = 5
    timed_steps = 20

    model = make_vit(patch_size).to(device)
    model.eval()

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Warmup steps
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []

    # Timed steps
    with torch.no_grad():
        for _ in range(timed_steps):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times_ms.append((end - start) * 1000)

    times_tensor = torch.tensor(times_ms)
    num_patches = (image_size // patch_size) ** 2

    return {
        "patch_size": patch_size,
        "num_patches": num_patches,
        "mean_ms": times_tensor.mean().item(),
        "std_ms": times_tensor.std(unbiased=True).item(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    for patch_size in [8, 16, 32]:
        print(f"\nBenchmarking patch size P={patch_size}...")
        result = benchmark_patch_size(patch_size, device)
        results.append(result)

        print(
            f"P={result['patch_size']}, "
            f"N={result['num_patches']}, "
            f"time={result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms"
        )

    df = pd.DataFrame(results)

    print("\nResults:")
    print(df.to_string(index=False))

    print("\nLaTeX table rows:")
    for r in results:
        print(
            f"{r['patch_size']} & {r['num_patches']} & "
            f"${r['mean_ms']:.3f} \\pm {r['std_ms']:.3f}$ ms \\\\"
        )


if __name__ == "__main__":
    main()
