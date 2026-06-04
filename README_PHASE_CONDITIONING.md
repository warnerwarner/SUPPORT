# --use_phase_conditioning Flag Documentation

## Overview
Complete documentation of the `--use_phase_conditioning` flag implementation in the SUPPORT denoising model. This flag enables the model to leverage sinusoidal phase information extracted from raw position signals in zarr data files.

## Documents in This Suite

### 1. PHASE_CONDITIONING_SUMMARY.md (Executive Summary)
**Best for:** Quick understanding of the overall flow and architecture
- 5-stage pipeline overview
- Key implementation details
- Discovered issues and fixes
- Testing checklist
- Critical code paths
- Recommended usage examples

**Start here if:** You want a high-level understanding in 15 minutes

### 2. PHASE_CONDITIONING_TRACE.md (Detailed Technical Trace)
**Best for:** Complete understanding of every code path
- 15 major sections covering all components
- File references with exact line numbers
- Data structures and tensor shapes at each stage
- Both CPU and GPU phase extraction
- Forward pass implementation details
- Random transform handling with phase
- Distributed training considerations

**Start here if:** You need to understand the implementation in depth or debug issues

### 3. PHASE_CONDITIONING_QUICK_REF.md (Quick Reference)
**Best for:** Fast lookup while coding or debugging
- File locations summary table
- Data flow overview diagram
- Key shapes throughout pipeline
- Conditional logic examples
- Critical dependencies
- Common issues and debugging
- Testing code snippets

**Start here if:** You need to quickly find where something is or check a specific detail

## Quick Start

### Command Line Usage
```bash
# Correct usage (train_distributed.py)
python src/train_distributed.py \
    --use_phase_conditioning \
    --is_zarr \
    --is_raw \
    --noisy_data /path/to/zarr/files \
    --batch_size 8 \
    --n_epochs 100

# Note: train.py has missing parameters - use train_distributed.py
```

### What the Flag Does

1. **Extracts phase** from `store["position"]` in zarr files
2. **Converts to sin/cos** for numerical stability
3. **Augments input** with phase information at each frame/row
4. **Modifies architecture:**
   - U-Net input: 3×(T-1) channels instead of T-1
   - BS-Net: 3 channels at first layer instead of 1
   - Learned projection layers for phase feature re-injection

## Data Format Changes

### Without --use_phase_conditioning
```python
batch = (image, coords, ds_idx, mean, std)  # 5-tuple
model(image)  # No phase arguments
```

### With --use_phase_conditioning
```python
batch = (image, coords, ds_idx, mean, std, phase_sin, phase_cos)  # 7-tuple
model(image, phase_sin, phase_cos)  # With phase arguments
```

## Architecture Changes

### U-Net
- **Without phase:** T-1 input channels (images only)
- **With phase:** 3×(T-1) input channels (images + sin + cos)

### BS-Net First Layer
- **Without phase:** 1 input channel (center frame)
- **With phase:** 3 input channels (center frame + sin + cos)

### BS-Net Deeper Layers
- **Without phase:** Scalar multiplication for feature injection
- **With phase:** Learned 1×1 convolutions for feature projection

## Key Tensor Shapes

| Stage | Shape | Notes |
|-------|-------|-------|
| Position signal (zarr) | (T, H, W) | Sinusoid along width |
| Extracted phase | (T, H) | Per-frame, per-row |
| Batch phase | (B, T, H) | Batched patches |
| U-Net phase (broadcast) | (B, T-1, H, W) | Expanded to spatial dims |
| BS-Net phase | (B, H) | Center frame only |
| U-Net input | (B, 3×(T-1), H, W) | [image, sin, cos] concatenated |
| BS-Net input | (B, 3, H, W) | [image, sin, cos] concatenated |

## Phase Extraction Method

The flag uses Hilbert transform to extract phase:

1. Load position signal: (T, H, W)
2. Compute analytic signal along W dimension
3. Extract phase angle at center pixel: phase = angle(analytic[:, :, W//2])
4. Convert to sin and cos for stability

## Requirements

- **Zarr files** with `store["position"]` data
- **Raw data** (--is_raw flag required)
- **Position signal shape:** (T_total, H_total, W_total)

## Known Issues

### Issue 1: train.py Doesn't Pass Phase
**File:** src/train.py lines 349-357, 388-401
- Missing `use_phase_conditioning` parameter to gen_train_dataloader()
- Missing `use_phase_conditioning` parameter to SUPPORT()
- **Workaround:** Use train_distributed.py instead

### Issue 2: Rotations Disabled With Phase
- 90° rotations are automatically disabled when phase is active
- Reason: Phase is per-row; rotation would mix row/column dimensions
- This is expected behavior, not a bug

### Issue 3: DDP find_unused_parameters
- With phase conditioning, must set find_unused_parameters=True
- Prevents errors about unused projection layers
- This is handled correctly in train_distributed.py

## Testing

### Verify Phase Extraction
```python
from src.utils.dataset import extract_phase
import zarr

store = zarr.open('/path/to/data.zarr')
position_signal = store["position"][:]  # (T, H, W)
phase_sin, phase_cos = extract_phase(position_signal)
print(phase_sin.shape)  # Should be (T, H)
```

### Verify Batch Format
```python
from src.utils.dataset import gen_train_dataloader

dataloader = gen_train_dataloader(
    ...,
    use_phase_conditioning=True,
    is_zarr=True,
    is_raw=True
)
batch = next(iter(dataloader))
print(len(batch))  # Should be 7 (with phase)
```

### Verify Model Forward Pass
```python
from model.SUPPORT import SUPPORT
import torch

model = SUPPORT(..., use_phase_conditioning=True)
x = torch.randn(B, T, H, W)
phase_sin = torch.randn(B, T, H)
phase_cos = torch.randn(B, T, H)
output = model(x, phase_sin, phase_cos)
print(output.shape)  # Should be (B, 1, H, W)
```

## Performance Considerations

### Memory
- Input channels triple for U-Net: 3×(T-1) instead of T-1
- May require smaller batch sizes
- Phase tensors are (B, T, H) compared to images (B, T, H, W)

### Computation
- Phase extraction via Hilbert transform happens once per dataset
- Additional 1×1 convolutions in BS-Net
- Similar forward pass time to baseline

### Training
- Phase provides spatial context → potentially faster convergence
- Rotations disabled → less data augmentation

## References

- Related test files:
  - `tests/test_phase_conditioning.py`
  - `tests/test_phase_structure.py`
  - `tests/test_real_data_phase.py`

- Implementation files:
  - `src/utils/util.py` - Argument definition
  - `src/utils/dataset.py` - Data loading and phase extraction
  - `src/train.py` - Training loop (incomplete)
  - `src/train_distributed.py` - Distributed training (complete)
  - `model/SUPPORT.py` - Model architecture

## Contact & Issues

For issues related to phase conditioning:
1. Check PHASE_CONDITIONING_QUICK_REF.md for common solutions
2. Review the detailed trace in PHASE_CONDITIONING_TRACE.md
3. Check test files for usage examples
4. Verify zarr files have `store["position"]` data
5. Ensure using `train_distributed.py` (not `train.py`)

## Document Metadata

- **Created:** April 27, 2026
- **Last Updated:** April 27, 2026
- **Scope:** Complete pipeline of --use_phase_conditioning flag
- **Files Analyzed:** 5 major files, 70+ code locations
- **Status:** Production-ready implementation with minor issues in train.py

---

## Navigation

- **Quick Start?** → Read PHASE_CONDITIONING_SUMMARY.md
- **Deep Dive?** → Read PHASE_CONDITIONING_TRACE.md
- **Quick Lookup?** → Use PHASE_CONDITIONING_QUICK_REF.md
- **Want to Know Everything?** → Start here, then read in order

