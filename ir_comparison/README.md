# IR Comparison: Torch-MLIR vs BladeDISC

Generates and inspects MLIR intermediate representations from two compilers
for an identical BERT-like transformer model.

## Structure

```
scripts/
  model_def.py            — shared model (2-layer, hidden=64, BERT-like structure)
  1_dump_torch_mlir.py    — generates stablehlo MLIR via Torch-MLIR
  2_dump_bladedisc.py     — generates mhlo MLIR via BladeDISC
output/
  torch_mlir_stablehlo_dynamic_2L.mlir   — Torch-MLIR output (stablehlo dialect)
  bd2L_dump.*.pretty.mlir                — BladeDISC output (mhlo dialect)
```

## How to Run

**Build the Torch-MLIR container (once):**
```powershell
docker build -f Dockerfile.torch_mlir -t torch-mlir-env .
```

**Dump Torch-MLIR IR:**
```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" torch-mlir-env python /workspace/scripts/1_dump_torch_mlir.py
```

**Dump BladeDISC IR:**
```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" bladedisc/bladedisc:latest-runtime-torch1.12.0-cu113 python /workspace/scripts/2_dump_bladedisc.py
```

## IR Signatures

**Torch-MLIR** (`stablehlo` dialect):
```
func.func @main(%arg0: tensor<?x?x64xf32>) -> tensor<?x?x64xf32>
```

**BladeDISC** (`mhlo` dialect):
```
func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
```

Both use dynamic shapes (`?`). BladeDISC additionally makes the hidden dimension symbolic.
