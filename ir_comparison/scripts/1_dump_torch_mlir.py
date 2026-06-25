"""
1_dump_torch_mlir.py — dump Torch-MLIR stablehlo IR with dynamic shapes.

Container: torch-mlir-env  (PyTorch 2.5.1 + torch-mlir)
Output:    output/torch_mlir_stablehlo_dynamic_2L.mlir

Why 2 layers / hidden=64:
  torch.export with dynamic shapes runs symbolic shape analysis through every
  layer. On a 12-layer BERT this hangs (>1 hour). A 2-layer model completes
  in ~30 seconds with identical IR structure — just fewer repeated blocks.

Why batch=2 for the export dummy:
  torch.export treats size=1 as a compile-time constant (special broadcast
  case). Using batch=2 forces the exporter to keep the dimension symbolic.
  The resulting IR has tensor<?x?x64xf32> — genuinely dynamic.
"""
import sys
sys.path.insert(0, "/workspace/scripts")

import torch
from torch.export import Dim
import torch_mlir.fx as fx
from model_def import SimpleBert

torch.manual_seed(42)
model = SimpleBert(layers=2, hidden=64, heads=8).eval()

# batch=2 so the exporter sees a non-trivial value and keeps the dim symbolic
dummy = torch.randn(2, 16, 64)
batch = Dim("batch", min=2, max=64)
seq   = Dim("seq",   min=1, max=512)

print(f"PyTorch: {torch.__version__}")
print(f"Input:   {dummy.shape}")
print("Exporting to stablehlo (dynamic shapes)...")

exported = torch.export.export(
    model,
    (dummy,),
    dynamic_shapes={"x": {0: batch, 1: seq}},
    strict=False,
)
print(f"  Graph nodes: {len(list(exported.graph.nodes))}")

mlir_module = fx.export_and_import(exported, output_type=fx.OutputType.STABLEHLO)
mlir_text   = mlir_module.operation.get_asm(large_elements_limit=10)

out = "/workspace/output/torch_mlir_stablehlo_dynamic_2L.mlir"
with open(out, "w") as f:
    f.write(mlir_text)

print(f"  Saved: {out}  ({len(mlir_text):,} chars)")
print(f"  Dynamic dims (?): {mlir_text.count('?')}")
print(f"  Signature: {mlir_text.split(chr(10))[1].strip()}")
