"""
2_dump_bladedisc.py — dump BladeDISC mhlo IR + full compilation log.

Container: bladedisc/bladedisc:latest-runtime-torch1.12.0-cu113
Output:    output/bd2L_dump.*.pretty.mlir   — pre-fusion mhlo IR (readable)
           output/bd2L_mhlo_compile.*.log   — full pipeline log with post-fusion
                                              lhlo IR and ral_kernel_launch count

DISC_DEBUG=true is the key flag — without it only the pre-fusion mhlo is dumped.
With it, the full compilation pipeline (mhlo → bufferize → lhlo → fusion → LLVM)
is logged. The ral_kernel_launch calls in the LLVM section count actual GPU kernels.
"""
import os, sys, shutil, glob
os.environ["TORCH_BLADE_DEBUG_LOG"] = "true"   # dumps pre-fusion mhlo pretty.mlir
os.environ["DISC_DEBUG"]            = "true"   # dumps full pipeline log (post-fusion)

sys.path.insert(0, "/workspace/scripts")
import torch
import torch_blade
from model_def import SimpleBert

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SimpleBert(layers=2, hidden=64, heads=8).eval().to(device)
dummy  = torch.randn(1, 16, 64, device=device)

print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
print(f"Input:   {dummy.shape}")

traced = torch.jit.trace(model, dummy, strict=False).eval()

print("Running BladeDISC optimization...")
with torch.no_grad(), torch_blade.Config():
    optimized = torch_blade.optimize(traced, allow_tracing=True, model_inputs=(dummy,))
print("  Done.")

# Copy all dumped files to output/
out_dir = "/workspace/output"
for f in glob.glob("dump_dir/**/*", recursive=True):
    if os.path.isfile(f):
        dest = os.path.join(out_dir, f"bd2L_{os.path.basename(f)}")
        shutil.copy2(f, dest)
        print(f"  Copied: {os.path.basename(f)}  ({os.path.getsize(dest)//1024} KB)")

# Quick post-fusion metric from the log
log = next(glob.glob(os.path.join(out_dir, "bd2L_mhlo_compile*.log")), None)
if log:
    count = sum(1 for line in open(log) if "ral_kernel_launch" in line)
    print(f"\nPost-fusion GPU kernel launches (ral_kernel_launch in LLVM IR): {count}")

# Correctness check
with torch.no_grad():
    ok = torch.allclose(model(dummy), optimized(dummy), atol=1e-2)
    print(f"Output correctness: {'OK' if ok else 'MISMATCH'}")
