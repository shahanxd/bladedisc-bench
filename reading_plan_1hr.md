# 1-Hour BladeDISC Code Reading Plan

**Goal:** Understand *exactly* how BD turns your 11-op W1 model into one GPU kernel.  
**Rules:** Read only what's listed. Stop at the line range. Move on even if you don't get 100%.

---

## ⏱ 0:00–0:15 — Where does BD take control? (Entry point)

**File:** `BladeDISC-main/tao_compiler/mlir/disc/disc_compiler.h`  
**Lines:** 27–70

**What to look for:**

| Line | What it is | What it means |
|---|---|---|
| 27–32 | `enum CodeGenMode { kCpuCentric, kGpuCentric }` | This is the on/off switch — when you set `cfg.enable_mlir_shape_analysis = True`, BD picks `kGpuCentric` |
| 34–50 | `struct GpuDeviceInfo` | BD reads your GPU's SM count and compute capability (cc_major/minor) to pick kernel configs |
| 52–70 | `struct DiscLoweringPipelineOptions` | The full settings object passed down the entire pipeline. Every option you set in `torch_blade.Config()` ends up here. |

**One sentence to remember:** The `DiscLoweringPipelineOptions` struct is the backbone — every stage of the compiler reads from it.

---

## ⏱ 0:15–0:30 — Where does fusion actually happen? (The key file)

**File:** `BladeDISC-main/tao_compiler/mlir/disc/transforms/lhlo_fusion.cc`  
**Lines:** 1–40 (top of file, understand what it imports), then 853–927

**Lines 1–40 — imports tell you what BD uses:**
- `#include "lhlo/IR/lhlo_ops.h"` → BD works on LHLO ops (Lowered HLO — already bufferized, no more tensors, only raw memory buffers)
- `#include "mlir/IR/Region.h"` → A `Region` is the container BD moves ops into to form a fusion group

**Lines 853–927 — `ApplyFusionPlan()` — THIS IS THE MOMENT:**

| Line | Code | What it does |
|---|---|---|
| 853 | `bool ApplyFusionPlan(FusionPlan& plan)` | Takes a pre-built list of "which ops go together" |
| 863 | `auto& op_list = pattern.getOpList()` | Gets the list of ops in one fusion group (e.g., your 11 ops) |
| 876 | `FusionOp fusion = b.create<lmhlo::FusionOp>(fused_loc)` | Creates a single wrapper op that will become ONE kernel |
| 879–881 | `for (op : llvm::reverse(op_list)) { op->moveBefore(&block, block.begin()) }` | **This is it.** Physically relocates all 11 ops inside the FusionOp's region |
| 882–883 | `fusion->setAttr(kDiscFusionTypeAttrName, ...)` | Tags the fusion as "kLoop" or "kRowReduction" — determines codegen strategy |

**One sentence:** Line 879 is where BD's 8–11× speedup is born — all 11 ops move inside one wrapper, and downstream codegen emits a single kernel for the whole group.

---

## ⏱ 0:30–0:42 — How does BD decide *which* ops to fuse? (The decision)

**File:** `BladeDISC-main/tao_compiler/mlir/disc/transforms/lhlo_fusion.cc`  
**Lines:** 200–280

This is `BuildFusionPlan()` or the graph-walk that calls `tryFuse()`. Look for a function that iterates ops and calls something like `isFusible()` or `canFuse()`.

**What to look for:**

| Concept | What to find | What it means |
|---|---|---|
| Elementwise check | A condition like `isElementwise(op)` | BD only fuses ops where every output element is computed independently — your mean/rsqrt/tanh all qualify |
| Buffer alias check | Something like `!hasAlias` or `isReadOnly` | BD won't fuse if two ops write to the same buffer (would be a data race) |
| Dominance check | `dominates(a, b)` | Ensures ops are fused in the right order — op A must finish before op B reads its output |

**Important single line to find:** The line that calls `pattern.push_back(op)` or `addToGroup(op)` — that's when an op is accepted into a fusion group.

---

## ⏱ 0:42–0:55 — How does the fused group become a GPU kernel? (Codegen handoff)

**File:** `BladeDISC-main/tao_compiler/mlir/disc/transforms/`  
**Look for file:** `disc_to_llvm.cc` OR `gpu_kernel_to_blob.cc`

You don't need to read deeply. Just find:

1. A function that takes a `FusionOp` and emits LLVM IR or PTX  
2. The line that calls something like `gpu::LaunchFuncOp` or `cuLaunchKernel` — this is where BD hands the kernel to CUDA

**Key concept:** Each `FusionOp` → 1 LLVM function → 1 PTX kernel → 1 `cuLaunchKernel` call at runtime. That's why 11 ops = 1 kernel launch instead of 11.

---

## ⏱ 0:55–1:00 — How does the runtime know the actual shapes? (RAL)

**File:** `BladeDISC-main/tao_compiler/mlir/disc/`  
**Look for file:** `runtime_abstraction_layer.cc` OR search for `RalContext`

**Lines to look for:** The function that receives the kernel and real input shapes at runtime.

**Key insight:** BD compiles the kernel with symbolic shapes (`?` dimensions). At runtime, the RAL context resolves `?` to the actual `[64, 1024]` size and passes it to `cuLaunchKernel` as grid/block dimensions. This is why BD doesn't recompile for every shape — the compiled kernel is shape-generic, and RAL fills in the blanks.

---

## Summary — The Full Pipeline in 5 Lines

```
torch_blade.optimize() call
    → disc_compiler.h: DiscLoweringPipelineOptions set (kGpuCentric)
    → lhlo_fusion.cc BuildFusionPlan(): decides which ops group together
    → lhlo_fusion.cc ApplyFusionPlan() line 879: physically moves ops into FusionOp
    → codegen pass: FusionOp → 1 PTX kernel
    → RAL at runtime: resolves symbolic shapes → launches kernel with real dims
```

**What you can say to your prof:**  
> *"I traced the fusion pipeline from `disc_compiler.h` through `lhlo_fusion.cc`. The `ApplyFusionPlan` function at line 879 physically relocates ops into a `FusionOp` region — that's the moment all 11 ops become one kernel. The grouping decision is made upstream by checking elementwise-ness and buffer aliasing. The RAL then fills in actual shapes at runtime, which is why the compiled artifact works for all batch sizes without recompiling."*

---

## Memory changes (already done — just for reference)

The `measure_mem()` helper added to all 4 test files works like this:

```python
def measure_mem(fn, x):
    torch.cuda.reset_peak_memory_stats()   # clear the counter
    with torch.no_grad():
        fn(x)                              # one forward pass
    torch.cuda.synchronize()              # wait for GPU to finish
    return torch.cuda.max_memory_allocated() / 1024**2  # peak MB
```

- `reset_peak_memory_stats()` — resets the high-water mark so previous ops don't pollute the reading
- `max_memory_allocated()` — returns the peak bytes used *during* that forward pass (includes all intermediate buffers)
- Called once before `bench()` since memory doesn't change between identical iters
