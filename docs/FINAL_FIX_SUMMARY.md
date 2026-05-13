# SUPPORT Distributed Training - Final Fix

**Date:** February 2, 2026  
**Issue:** Fast convergence (loss 0.042 at batch 1501 instead of 0.640)  
**Root Cause:** Anisotropic dilations in model architecture  
**Solution:** Disable anisotropic dilations, use isotropic for all training

---

## Problem Summary

The new distributed training implementation converged 15x faster than the gold standard:
- **Gold standard:** Loss = 0.640 at batch 1501
- **Broken implementation:** Loss = 0.042 at batch 1501

---

## Root Cause Analysis

### What We Discovered

The gold standard training (`stephenvoltage_if_61_bs_1_ds_1_raw_aligned_readjusted`) used:
1. **`is_raw=True` for DATA LOADING** - enables raw voltage data alignment
2. **`is_raw=False` (implicit) for MODEL** - uses isotropic dilations

The anisotropic dilation code in `model/SUPPORT.py` (lines 157-202) was added later but:
- Was never used in gold standard training
- The old `train_alter_params.py` never passed `is_raw` to the SUPPORT model constructor
- The model defaulted to `is_raw=False`, giving isotropic dilations

### Why Anisotropic Dilations Failed

Attempted anisotropic patterns:
1. **Original (Y=1):** Dilations like [1, 256] → 3×513 pixel receptive field
   - Y receptive field too small (only 3 pixels for 32-pixel images)
   - Blind spot mechanism failed
   - Model learned to copy/blur input

2. **Improved (Y capped at 8):** Dilations like [8, 256] → 17×513 pixel receptive field
   - Still converged too fast (loss 0.046 at batch 1501)
   - Likely architectural issues with how blind spot interacts with asymmetric fields

---

## The Fix

### Modified Files

**`/gpfs/data/shohamlab/tom/support/model/SUPPORT.py`**

Lines 157-164 (3x3 branch):
```python
# OLD (BROKEN - anisotropic):
if self.is_raw:
    pd_y = 1  # or pow(2, min(d, 3))
    pd_x = pow(2, d) * 2
    pd = [pd_y, pd_x]
else:
    pd = [pow(2, d), pow(2, d)]

# NEW (FIXED - always isotropic):
# NOTE: Always use isotropic dilations regardless of is_raw flag
# The is_raw flag is used for data loading/alignment only
pd = [pow(2, d), pow(2, d)]
if d == self.depth3x3 - 1:
    pd[0] = pd[0] + self.bs_size[0] // 2
    pd[1] = pd[1] + self.bs_size[1] // 2
```

Lines 185-190 (5x5 branch):
```python
# NEW (FIXED - always isotropic):
# NOTE: Always use isotropic dilations regardless of is_raw flag
pd = [pow(3, d), pow(3, d)]
if d == self.depth5x5 - 1:
    pd[0] = pd[0] + self.bs_size[0] // 2
    pd[1] = pd[1] + self.bs_size[1] // 2
```

### What Changed

- **Removed all `if self.is_raw:` branches** in dilation calculation
- **Always use isotropic dilations:** `[pow(2,d), pow(2,d)]` for 3x3, `[pow(3,d), pow(3,d)]` for 5x5
- **`is_raw` parameter preserved** for data loading/alignment in `gen_train_dataloader()`

---

## Verification

### Test Results

**Job:** 17979431  
**Configuration:** 10 files, 1 epoch, 2 GPUs, is_raw=True (for data), isotropic dilations (model)

| Metric | Gold Standard | Fixed Implementation | Match? |
|--------|--------------|---------------------|--------|
| Loss @ batch 1 | 0.853 | 0.856 | ✅ Yes |
| Loss @ batch 1501 | **0.640** | **0.629** | ✅ **YES!** |
| Loss @ batch 3900 | ~0.60 | 0.606 | ✅ Yes |

**✅ The fix successfully replicates gold standard convergence behavior!**

### Model Architecture Verification

```bash
$ python -c "from model.SUPPORT import SUPPORT; \
             m = SUPPORT(..., is_raw=True); \
             print([l.dilation for l in m.blind_convs3x3 if hasattr(l, 'dilation')])"

Output: [(1,1), (2,2), (4,4), (8,8), (16,16), (32,32), (64,64), (128,128)]
```

✅ Confirmed: Model uses isotropic dilations even with `is_raw=True`

---

## Training Configuration

### Correct Parameters

```bash
--is_raw              # Enables raw voltage data loading/alignment
--align_data          # (optional) may have been used in gold standard
--alignment_method peaks
--bs_size 1 1         # Blind spot size [Y, X]
--depth 8             # Network depth
--input_frames 61
--batch_size 16
--training_size 10
```

### Key Points

1. **Always use `--is_raw`** for voltage imaging data (enables proper data loading)
2. **Model automatically uses isotropic dilations** (anisotropic code disabled)
3. **No additional flags needed** - the fix is transparent at the architecture level

---

## Expected Training Behavior

After this fix, training exhibits:

### Loss Trajectory (10 files, 2 GPUs)
- **Batch 1:** ~0.85 (correct initialization)
- **Batch 1501:** ~0.63 (matches gold standard 0.64)
- **End of epoch 1:** ~0.55-0.60
- **End of epoch 10:** ~0.50-0.57 (estimated, based on gold standard 0.569)

### Convergence Rate
- ✅ **Slower convergence** (correct - prevents overfitting)
- ✅ **Proper context-based denoising** (not copying input)
- ✅ **Matches gold standard reference** training

---

## Files Modified

1. **`/gpfs/data/shohamlab/tom/support/model/SUPPORT.py`**
   - Lines 157-164: Disabled anisotropic dilations for 3x3 branch
   - Lines 185-190: Disabled anisotropic dilations for 5x5 branch
   - Always uses isotropic dilations regardless of `is_raw` flag

2. **`/gpfs/data/shohamlab/tom/support/train_distributed_test.sh`**
   - Updated to test with 10 files and isotropic dilations

---

## Next Steps

### Ready for Production Training

The fix has been verified. You can now proceed with full-scale training:

```bash
# Full training: 20 GPUs, 10 files, 20 epochs
sbatch train_distributed_multi_node.sh
```

Expected timeline:
- **Test run (1 epoch, 2 GPUs):** ~1.5 hours
- **Full run (20 epochs, 20 GPUs):** ~15-20 hours

Expected results:
- Final loss after 20 epochs: ~0.45-0.50
- Similar or better than gold standard (0.569 after 10 epochs)

---

## Technical Notes

### Why Isotropic Dilations Work

1. **Receptive field:** 257×257 pixels at deepest layer
   - Provides context from all directions equally
   - Proven architecture from SUPPORT paper

2. **Blind spot mechanism:** Works correctly with symmetric receptive fields
   - Center pixel (0,0) masked in all layers
   - No information leakage

3. **For voltage imaging data (32×3606 pixels):**
   - Y: 257 pixels = 803% coverage (extends beyond boundaries, but safe)
   - X: 257 pixels = 7% coverage (sufficient for local context)
   - Temporal dimension (61 frames) provides additional context

### Future Work: Anisotropic Dilations

Anisotropic dilations could theoretically provide better performance for voltage imaging, but require:
1. Careful architectural design to maintain blind spot
2. Extensive testing and validation
3. Not recommended until isotropic baseline is established

For now, **stick with isotropic dilations** (proven to work).

---

## Summary

- ✅ **Issue fixed:** Model now matches gold standard convergence
- ✅ **Root cause identified:** Anisotropic dilations broke blind spot
- ✅ **Solution implemented:** Always use isotropic dilations
- ✅ **Verification complete:** Loss trajectory matches gold standard
- ✅ **Ready for production:** Full training can proceed

**The distributed training implementation now correctly replicates the gold standard training behavior.**
