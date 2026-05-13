# SUPPORT Model Capacity Increase - 1.5x Width

**Date:** 2026-02-02  
**Purpose:** Increase model capacity to reduce high-frequency spatial noise that remains after training plateau

---

## Problem Statement

Training with the baseline model (depth=8, 64 channels) plateaus at loss ~0.5-0.6 after 10 epochs, with **high-frequency spatial noise remaining** in the denoised output. Validation loss tracks training loss (not overfitting), suggesting the model lacks sufficient capacity to capture fine spatial details.

---

## Solution: 1.5x Width Increase

### New Training Script

**File:** `/gpfs/data/shohamlab/tom/support/train_distributed_multi_node_wider.sh`

### Architecture Changes

| Parameter | Baseline | 1.5x Wider | Change |
|-----------|----------|------------|--------|
| `blind_conv_channels` | 64 | 96 | +50% |
| `unet_channels` | [64, 128, 256, 512, 1024] | [96, 192, 384, 768, 1536] | +50% each level |
| `one_by_one_channels` | [32, 16] | [48, 24] | +50% |
| `last_layer_channels` | [64, 32, 16] | [96, 48, 24] | +50% |
| `depth` | 8 | 8 | *unchanged* |
| `bs_size` | [1, 1] | [1, 1] | *unchanged* |

**Estimated Parameters:**
- Baseline: ~1.8M parameters
- 1.5x Wider: ~4.5-5.5M parameters (~2.5-3x increase)

### Training Parameters (Unchanged)

```bash
--batch_size 16           # May reduce to 12/8 if memory issues
--patch_size 61 16 320
--patch_interval 1 4 160
--n_epochs 20
--training_size 10
--depth 8
--bs_size 1 1
--is_raw
--use_amp
```

### Expected Outcomes

1. **Reduced high-frequency spatial noise** in denoised output
2. **Lower final loss:** Target <0.45 (vs. baseline plateau at 0.5-0.6)
3. **Continued improvement past epoch 10** (breaking through plateau)
4. **Training time:** 2-3x longer (~2-3 days vs. 1-1.5 days for baseline)

---

## How to Launch

```bash
cd /gpfs/data/shohamlab/tom/support
sbatch train_distributed_multi_node_wider.sh
```

**Resources:**
- 5 nodes × 4 GPUs = 20 GPUs (V100 32GB each)
- 4 days max walltime (should complete in 2-3 days)
- Experiment name: `stephenvoltage_if_61_bs_1_wider_1.5x_20epochs`

**Logs:**
- Job output: `/gpfs/home/warnet02/shohamlab/tom/tmp/<JOB_ID>.log`
- Job errors: `/gpfs/home/warnet02/shohamlab/tom/tmp/<JOB_ID>.err`
- Training log: `/gpfs/data/shohamlab/tom/support/results/logs/stephenvoltage_if_61_bs_1_wider_1.5x_20epochs.log`

**Results:**
- Checkpoints: `/gpfs/data/shohamlab/tom/support/results/checkpoints/stephenvoltage_if_61_bs_1_wider_1.5x_20epochs/`
- Samples: `/gpfs/data/shohamlab/tom/support/results/denoised_samples/stephenvoltage_if_61_bs_1_wider_1.5x_20epochs/`

---

## Monitoring Progress

### Check job status
```bash
squeue -u warnet02
```

### Monitor training log
```bash
tail -f /gpfs/data/shohamlab/tom/support/results/logs/stephenvoltage_if_61_bs_1_wider_1.5x_20epochs.log
```

### Watch for key metrics
- **Batch 1501 loss:** Should be ~0.58-0.63 (matching baseline)
- **Epoch 10 loss:** Should continue decreasing (not plateau)
- **Final loss (epoch 20):** Target <0.45

### GPU memory check
```bash
grep -i "memory\|oom" /gpfs/home/warnet02/shohamlab/tom/tmp/<JOB_ID>.err
```

If OOM errors occur:
1. Cancel job: `scancel <JOB_ID>`
2. Edit script: change `--batch_size 16` to `--batch_size 8`
3. Resubmit: `sbatch train_distributed_multi_node_wider.sh`

---

## Comparison with Baseline

### Baseline (Current)
- **Script:** `train_distributed_multi_node.sh`
- **Exp name:** `stephenvoltage_if_61_bs_1_bigger_context_window_multi_node`
- **Channels:** 64 baseline
- **Parameters:** ~1.8M
- **Plateau:** Loss ~0.5-0.6 after 10 epochs
- **Issue:** High-frequency spatial noise remains

### 1.5x Wider (New)
- **Script:** `train_distributed_multi_node_wider.sh`
- **Exp name:** `stephenvoltage_if_61_bs_1_wider_1.5x_20epochs`
- **Channels:** 1.5x increase across all layers
- **Parameters:** ~4.5-5.5M
- **Expected:** Lower final loss, reduced spatial noise
- **Trade-off:** 2-3x longer training time

---

## Next Steps (If Results Insufficient)

### Phase 2: Full 2x Width Increase

If 1.5x shows improvement but high-frequency noise still remains:

**Create:** `train_distributed_multi_node_wider_2x.sh`

**Changes:**
```bash
--blind_conv_channels 128         # 64 → 128 (2x)
--unet_channels 128 256 512 1024 2048
--one_by_one_channels 64 32
--last_layer_channels 128 64 32
--batch_size 8                    # Reduce for memory safety
```

**Expected:**
- ~4x parameters (~7-8M)
- Maximum capacity boost
- Training ~2.5-3x slower than baseline

### Alternative Approaches

1. **Increase depth:** `--depth 10` (larger receptive field)
2. **Larger patches:** `--patch_size 61 32 640` (more spatial context)
3. **More training data:** `--training_size 20` (if more files available)
4. **Hybrid:** Combine 1.5x width + depth increase

---

## Technical Notes

### Why This Should Work

1. **High-frequency spatial noise** indicates model lacks capacity for fine details
2. **Blind-spot branch (96 channels)** → better spatial feature extraction
3. **U-Net branch (1.5x channels)** → richer temporal context encoding
4. **Uniform scaling** ensures balanced capacity increase across architecture
5. **Conservative approach** minimizes overfitting risk (validation tracks training)

### Memory Considerations

- **V100 GPUs:** 32GB each (plenty of headroom)
- **Current batch_size=16:** Uses ~15-20GB per GPU (estimated)
- **1.5x model:** Should fit comfortably with batch_size=16
- **Safety margin:** Can reduce to batch_size=8 if needed without major impact

### Training Dynamics

- **Automatic Mixed Precision (AMP):** Enabled (`--use_amp`)
- **Learning rate:** 5e-4 (default Adam, same as baseline)
- **Loss function:** 0.5 × L1 + 0.5 × L2 (same as baseline)
- **Optimizer:** Adam (default)

---

## File Structure

```
/gpfs/data/shohamlab/tom/support/
├── train_distributed_multi_node.sh           # Baseline script
├── train_distributed_multi_node_wider.sh     # NEW: 1.5x width script
├── train_distributed_test.sh                 # Baseline test (2 GPU)
├── model/SUPPORT.py                          # Model (no changes needed)
├── src/train_distributed.py                  # Training loop (no changes needed)
├── src/utils/util.py                         # Config parser (supports new params)
├── FINAL_FIX_SUMMARY.md                      # Blind-spot fix documentation
└── WIDER_MODEL_README.md                     # This file
```

---

## Questions?

- **GPU memory issues?** → Reduce batch_size to 8
- **Still too slow?** → Consider smaller width increase or fewer epochs
- **Still not enough capacity?** → Proceed to Phase 2 (2x width)
- **Loss diverges?** → Check logs for numerical instability, may need lower LR

---

**Ready to launch when you are!** 🚀

```bash
sbatch train_distributed_multi_node_wider.sh
```
