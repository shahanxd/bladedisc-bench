import torch
import time

DEVICE = 'cuda'
DTYPE = torch.float16
HIDDEN_DIM = 1024
WARMUP = 50
ITERS = 200

SHAPES = [
    (1, HIDDEN_DIM),
    (8, HIDDEN_DIM),
    (32, HIDDEN_DIM),
    (64, HIDDEN_DIM),
    (128, HIDDEN_DIM),
    (256, HIDDEN_DIM),
]

class SimpleMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim * 4)
        self.act = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


def bench(fn, x, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn(x)
        t1.record()
        torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def measure_mem(fn, x):
    """Single forward pass to capture peak VRAM usage in MB."""
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def run():
    print("=" * 70)
    print("EXPERIMENT 6: The 'Happy Path' on Latest PyTorch - Eager vs PT2")
    print("=" * 70)

    model = SimpleMLP(HIDDEN_DIM).to(DEVICE, DTYPE).eval()
    
    print("1. Compiling with PyTorch Latest (torch.compile)...")
    pt2_model = torch.compile(model, dynamic=True)
    
    with torch.no_grad():
        pt2_model(torch.randn(1, HIDDEN_DIM, device=DEVICE, dtype=DTYPE))

    print("\n--- Benchmarking ---")
    print(f"{'Shape':>15}  {'Eager (ms)':>12}  {'PT2 (ms)':>12}")
    print("-" * 43)

    for shape in SHAPES:
        x = torch.randn(*shape, device=DEVICE, dtype=DTYPE)

        eager_mem_mb = measure_mem(model,     x)
        pt2_mem_mb   = measure_mem(pt2_model, x)

        eager_ms = bench(model,     x, WARMUP, ITERS)
        pt2_ms   = bench(pt2_model, x, WARMUP, ITERS)

        print(f"{str(shape):>15}  {eager_ms:>12.4f}  {pt2_ms:>12.4f}  "
              f"[mem eager={eager_mem_mb:.1f}MB  pt2={pt2_mem_mb:.1f}MB]")

if __name__ == '__main__':
    run()
