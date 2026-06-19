"""
Experiment 1: BladeDISC Wins — Memory-Bound Op Chain with Dynamic Shapes

What we test:
  A chain of ~10 individual elementwise + reduction ops that implement
  LayerNorm + GELU + residual manually (no pre-fused torch ops).

  In eager mode: each op is a separate CUDA kernel. ~10 global memory round-trips.
  With BladeDISC: fused into 1-2 kernels. One read, one write.

  We test across 8 different shapes to confirm BladeDISC handles
  dynamic shapes without recompiling each time.

BladeDISC API used:
  torch_blade.optimize(model, allow_tracing=True, model_inputs=(...))
"""

import torch
import torch_blade
import json
import os
import time

DEVICE = "cuda"
DTYPE = torch.float16
HIDDEN_DIM = 1024
WARMUP = 50
ITERS = 20
SCALE = 1

#SHAPES = [
#    (SCALE * 1,    HIDDEN_DIM),
#    (SCALE * 8,    HIDDEN_DIM),
#    (SCALE * 32,   HIDDEN_DIM),
#    (SCALE * 64,   HIDDEN_DIM),
#    (SCALE * 128,  HIDDEN_DIM),
#    (SCALE * 256,  HIDDEN_DIM),
#    (SCALE * 512,  HIDDEN_DIM),
#    (SCALE * 1024, HIDDEN_DIM),
#]

SHAPES = []

for i in range(1024):
    SHAPES.append((SCALE * i,    HIDDEN_DIM))

class ManualNormBlock(torch.nn.Module):
    """
    Manual LayerNorm + GELU + residual using individual ops.
    Each line = a separate kernel in eager mode.
    BladeDISC fuses them all into 1-2 kernels.
    """
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias   = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # --- Manual LayerNorm (~6 ops) ---
        mean     = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var      = (centered * centered).mean(dim=-1, keepdim=True)
        normed   = centered * torch.rsqrt(var + 1e-5)
        scaled   = normed * self.weight + self.bias

        # --- Manual GELU (~5 ops) ---
        x3     = scaled * scaled * scaled
        inner  = 0.7978845608 * (scaled + 0.044715 * x3)
        gelu   = 0.5 * scaled * (1.0 + torch.tanh(inner))

        # --- Residual add ---
        return x + gelu


def bench(fn, x, warmup, iters):
    with torch.no_grad():
        num_shapes = len(SHAPES)
        for _ in range(warmup):
            for i in range(num_shapes):
                fn(x[i])
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            for i in range(num_shapes):
                fn(x[i])
        t1.record()
        torch.cuda.synchronize()
    return t0.elapsed_time(t1) #/ iters


def measure_mem(fn, x):
    """Single forward pass to capture peak VRAM usage in MB."""
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def run():
    print("=" * 65)
    print("EXPERIMENT 1: BladeDISC Fusion — Memory-Bound Op Chain")
    print("=" * 65)
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {DTYPE} | Hidden: {HIDDEN_DIM}")
    print()

    model = ManualNormBlock(HIDDEN_DIM).to(DEVICE, DTYPE).eval()

    # ── BladeDISC compilation ──
    # Use a representative input for tracing.
    # BladeDISC's symbolic shape analysis makes the compiled
    # artifact valid for ALL shapes, not just the trace shape.
    trace_input = torch.randn(1, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)

    print("Compiling with BladeDISC... (first-time compilation, ~60-120s)")
    compile_start = time.time()
    with torch_blade.Config() as cfg:
        cfg.enable_mlir_shape_analysis = True   # symbolic shape propagation
        # Correct format: flat dict, each value is list-of-shapes (one per input tensor)
        # opts is a list-of-list-of-shapes (multiple optimal configs)
        cfg.dynamic_tuning_shapes = {
            "min":  [[1,    HIDDEN_DIM]],           # min shape for input_0
            "max":  [[1024 * SCALE, HIDDEN_DIM]],           # max shape for input_0
            "opts": [[[1, HIDDEN_DIM]], [[8, HIDDEN_DIM]], [[16, HIDDEN_DIM]]],  # opt shapes
        }
    opt_model = torch_blade.optimize(
        model,
        allow_tracing=True,
        model_inputs=(trace_input,),
    )
    compile_time = time.time() - compile_start
    print(f"Compilation done in {compile_time:.1f}s\n")

    opt_model1 = torch.compile(model, dynamic=True)
    eager_ms = opt_model1(trace_input)

    results = []
    print(f"{'Shape':>15}  {'Compile (ms)':>12} {'Eager (ms)':>12}  {'BladeDISC (ms)':>16}  {'Speedup':>8}")
    print("-" * 57)

    i = 0
    x = {}
    for shape in SHAPES:
        x[i] = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
        i = i + 1

    # Use mid-range shape for memory measurement
    mem_input = x[min(64, len(x) - 1)]
    eager_mem_mb   = measure_mem(model,      mem_input)
    compile_mem_mb = measure_mem(opt_model1, mem_input)
    blade_mem_mb   = measure_mem(opt_model,  mem_input)

    eager_ms   = bench(model,      x, WARMUP, ITERS)
    compile_ms = bench(opt_model1, x, WARMUP, ITERS)
    blade_ms   = bench(opt_model,  x, WARMUP, ITERS)
    speedup    = eager_ms / blade_ms

    results.append({
        "shape": list(shape),
        "compile_ms":      round(compile_ms, 4),
        "eager_ms":        round(eager_ms, 4),
        "bladedisc_ms":    round(blade_ms, 4),
        "speedup":         round(speedup, 2),
        "eager_mem_mb":    round(eager_mem_mb, 2),
        "compile_mem_mb":  round(compile_mem_mb, 2),
        "bladedisc_mem_mb": round(blade_mem_mb, 2),
    })
    print(f"{str(shape):>15} {compile_ms:>12.4f} {eager_ms:>12.4f}  {blade_ms:>16.4f}  {speedup:>7.2f}x  "
          f"[mem eager={eager_mem_mb:.1f}MB  compile={compile_mem_mb:.1f}MB  bd={blade_mem_mb:.1f}MB]")

    avg = sum(r["speedup"] for r in results) / len(results)
    mx  = max(r["speedup"] for r in results)
    print("-" * 57)
    print(f"Average speedup: {avg:.2f}x    Max: {mx:.2f}x")

    # Save
    out = {
        "experiment": "ex1_bladedisc_fusion",
        "description": "Manual LayerNorm+GELU+residual — ~10 ops fused by BladeDISC",
        "compiler": "BladeDISC torch_blade.optimize",
        "compile_time_s": round(compile_time, 1),
        "device": torch.cuda.get_device_name(0),
        "dtype": str(DTYPE),
        "results": results,
        "summary": {"avg_speedup": round(avg, 2), "max_speedup": round(mx, 2)},
    }
    os.makedirs("./results", exist_ok=True)
    with open("./results/ex1_bladedisc_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to /results/ex1_bladedisc_results.json")
    return out


if __name__ == "__main__":
    run()
