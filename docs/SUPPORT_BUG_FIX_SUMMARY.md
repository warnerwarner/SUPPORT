# SUPPORT Model Bug Fix Summary

## Overview

This document summarizes the bugs found in the SUPPORT model and related training scripts, along with the fixes applied.

## Bugs Found and Fixed

### 1. Typos in `use_phase_conditioning` Flag

**Files Affected:**
- `model/SUPPORT.py`
- `src/train_distributed.py`
- `train_distributed_multi_node.sh`

**Issue:**
- The SUPPORT model constructor used `use_phase_conditioning` as the parameter name
- But internally checked `self.use_phase_encoding` (a typo)
- The training script also used `--use_phase_encoding` instead of `--use_phase_conditioning`

**Fix:**
- Changed internal references from `use_phase_encoding` to `use_phase_conditioning` in:
  - `model/SUPPORT.py`: `_gen_bsnet()` and `forward_bsnet()` methods
  - `src/train_distributed.py`: Model instantiation
- Updated `train_distributed_multi_node.sh` to use `--use_phase_conditioning`

### 2. Dimension Mismatch in Phase Conditioning

**File Affected:** `model/SUPPORT.py`

**Issue:**
When using `use_phase_conditioning`, the code attempted to broadcast phase tensors `(B, T, H)` to match the UNet output `(B, C, H, W)`. However, the dimension ordering was incorrect, causing a runtime error:

```
RuntimeError: The expanded size of the tensor (320) must match the existing size (16) at non-singleton dimension 2.
```

**Fix:**
In `forward_bsnet()` method (line ~328), corrected the dimension unpacking and broadcasting:

```python
# Before (incorrect):
B, C, W, H = unet_out.shape
p_sin = phase_sin.unsqueeze(-1).expand(B, -1, H, W)

# After (correct):
B, C, H, W = unet_out.shape
p_sin = phase_sin.unsqueeze(-1).expand(B, -1, H, W)
```

### 3. CUDA Version Mismatch on gl40s Partition

**Issue:**
The `voltage_imaging` pixi environment has PyTorch built with CUDA 12.8, but the `gl40s` nodes have NVIDIA drivers supporting CUDA 13.0. This caused CUDA initialization errors on those nodes.

**Recommendation:**
Use `gpu4_short` or `gpu4_medium` partitions (V100 GPUs) which have compatible CUDA drivers, or update the environment to use a CUDA 13.0-compatible PyTorch version.

### 4. Data Path Issue

**Files Affected:**
- `train_distributed_multi_node.sh`
- `tests/train_distributed_test_gpu4.sh`

**Issue:**
The scripts used `data_dir="/gpfs/home/warnet02/data/stephen"` but the correct path is `/gpfs/data/shohamlab/tom/stephen`.

**Fix:**
Updated both scripts to use the correct path.

## Training Scripts

### New Scripts Created

1. **`tests/train_distributed_test_gpu4.sh`** - Small test script
   - Partition: `gpu4_short`
   - Nodes: 1, GPUs: 2
   - Epochs: 2, Training files: 3

2. **`train_distributed_multi_node_gpu4.sh`** - Full training script
   - Partition: `gpu4_short`
   - Nodes: 2, GPUs: 8 (4 per node)
   - Epochs: 20, Training files: 20

## Testing

Local testing confirmed the fixes work:
- Single forward pass: ✓ Success
- Multiple training loops with 3 files: ✓ Success

## Usage

To run training with phase conditioning:

```bash
# Test run
sbatch tests/train_distributed_test_gpu4.sh

# Full training
sbatch train_distributed_multi_node_gpu4.sh
```

Or use the original script (if data path is corrected):

```bash
sbatch train_distributed_multi_node.sh
```

## Notes

- The `--use_phase_conditioning` flag enables phase conditioning in the SUPPORT model
- This helps the model handle periodic artifacts by conditioning on the signal's phase
- The phase is extracted from the data and fed as sin/cos encodings to the model
