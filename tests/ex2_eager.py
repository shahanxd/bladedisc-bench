"""
Experiment 2: BladeDISC Barely Helps — Standalone Large MatMul (Compute-Bound)

What we test:
  A single torch.mm(A, B) dispatches directly to cuBLAS.
  cuBLAS is already hand-tuned for matrix multiplication.
  There is ONE op — nothing to fuse.
  The bottleneck is raw FLOPS (tensor cores), not memory bandwidth.

  BladeDISC (or any compiler) cannot improve this because:
    1. There is nothing to fuse
    2. cuBLAS already picks the optimal tiling/schedule
    3. The GPU tensor cores are already saturated

  Expected result: speedup ≈ 1.0x (within noise)
"""

import torch
import json
import os
import time

DEVICE = "cuda"
DTYPE  = torch.float16
WARMUP = 50
ITERS  = 2000

# (M, K, N) — A[M,K] x B[K,N] -> C[M,N]
SHAPES = [
    (128,  128,  128),
    (256,  256,  256),
    (512,  512,  512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1024, 4096, 1024),   # FFN-style: wide K
    (4096, 1024, 4096),   # FFN-style: wide output
]


class MatMulModule(torch.nn.Module):
    """Wraps a single torch.mm so torch_blade can trace it."""
    def forward(self, a, b):
        return torch.mm(a, b)


def bench(fn, a, b, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn(a, b)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn(a, b)
        t1.record()
        torch.cuda.synchronize()
    return t0.elapsed_time(t1) #/ iters


def measure_mem(fn, a, b):
    """Single forward pass to capture peak VRAM usage in MB."""
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn(a, b)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def run():
    print("=" * 70)
    print("EXPERIMENT 2: BladeDISC on Standalone MatMul — Compiler Barely Helps")
    print("=" * 70)
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {DTYPE}")
    print()

    model = MatMulModule().to(DEVICE).eval()

    # Use a representative shape for tracing
    trace_a = torch.randn(1024, 1024, device=DEVICE, dtype=DTYPE)
    trace_b = torch.randn(1024, 1024, device=DEVICE, dtype=DTYPE)

    print("Compiling with BladeDISC...")
    compile_start = time.time()
    opt_model = torch.compile(model)
    compile_time = time.time() - compile_start
    print(f"Compilation done in {compile_time:.1f}s\n")

    opt_model(trace_a, trace_b)

    results = []
    print(f"{'Shape (M,K,N)':>22}  {'Eager (ms)':>12}  {'BladeDISC (ms)':>16}  {'Speedup':>8}  {'TFLOPS':>8}")
    print("-" * 72)

    for (M, K, N) in SHAPES:
        a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)

        eager_mem_mb   = measure_mem(model,     a, b)
        compile_mem_mb = measure_mem(opt_model, a, b)

        eager_ms  = bench(model,     a, b, WARMUP, ITERS)
        blade_ms  = bench(opt_model, a, b, WARMUP, ITERS)
        speedup   = eager_ms / blade_ms
        tflops    = (2 * M * K * N) / (eager_ms / 1000) / 1e12

        results.append({
            "shape":          [M, K, N],
            "eager_ms":       round(eager_ms, 4),
            "bladedisc_ms":   round(blade_ms, 4),
            "speedup":        round(speedup, 2),
            "tflops_eager":   round(tflops, 2),
            "eager_mem_mb":   round(eager_mem_mb, 2),
            "compile_mem_mb": round(compile_mem_mb, 2),
        })
        print(f"  {str((M,K,N)):>20}  {eager_ms:>12.4f}  {blade_ms:>16.4f}  {speedup:>7.2f}x  {tflops:>7.2f}  "
              f"[mem eager={eager_mem_mb:.1f}MB  compile={compile_mem_mb:.1f}MB]")

    avg = sum(r["speedup"] for r in results) / len(results)
    print("-" * 72)
    print(f"Average speedup: {avg:.2f}x  (expected ~1.0x — no improvement)")

    out = {
        "experiment": "ex2_bladedisc_matmul",
        "description": "Standalone MatMul — compute-bound, cuBLAS already optimal",
        "compiler": "BladeDISC torch_blade.optimize",
        "compile_time_s": round(compile_time, 1),
        "device": torch.cuda.get_device_name(0),
        "dtype": str(DTYPE),
        "results": results,
        "summary": {"avg_speedup": round(avg, 2)},
    }
    os.makedirs("./results", exist_ok=True)
    with open("./results/ex2_bladedisc_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to /results/ex2_bladedisc_results.json")
    return out


if __name__ == "__main__":
    run()
