# --use_phase_conditioning Flag: Complete Flow Analysis

## Executive Summary

The `--use_phase_conditioning` flag enables the SUPPORT denoising model to leverage sinusoidal phase information extracted from raw position signals in zarr data files. The phase (represented as sin and cos values) provides per-row, per-frame spatial context that the model learns to use during denoising.

**Command Usage:**
```bash
python train_distributed.py --use_phase_conditioning --is_zarr --is_raw ...
```

---

## 5-Stage Pipeline

### Stage 1: Command-Line Argument
**File:** `src/utils/util.py` (lines 215-219)
- Boolean flag with `action="store_true"`
- Accessed as: `opt.use_phase_conditioning`
- Default: False (disabled unless explicitly specified)

### Stage 2: Data Loading & Phase Extraction
**File:** `src/utils/dataset.py` (lines 713-847)

**Flow:**
1. `gen_train_dataloader()` receives `use_phase_conditioning` parameter
2. If True: Initialize empty `phase_sin_list` and `phase_cos_list`
3. For each zarr file with `is_raw=True`:
   - Load `position_signal = store["position"][:]` (shape: T×H×W)
   - Call `extract_phase(position_signal)` using Hilbert transform
   - Output: `phase_sin, phase_cos` tensors (shape: T×H)
   - Append to respective lists
4. Pass phase lists to `DatasetSUPPORT` dataset class

**Phase Extraction Method (Hilbert Transform):**
- Compute analytic signal along W dimension (sinusoid axis)
- Extract phase angle at center pixel: `phase = angle(analytic[:,:,W//2])`
- Convert to sin/cos for numerical stability (avoids phase wrapping issues)
- Result: Phase per frame and row

### Stage 3: Batch Generation
**File:** `src/utils/dataset.py` (lines 523-620)

**DatasetSUPPORT.__getitem__() returns:**

**With phase conditioning:**
```python
(
    noisy_image,          # (T, H, W) 4D tensor
    patch_coordinates,    # Spatial indices
    dataset_index,
    noisy_image_avg,      # Per-file mean
    noisy_image_std,      # Per-file std
    phase_sin,            # (T_patch, H_patch) float32
    phase_cos,            # (T_patch, H_patch) float32
)  # 7-tuple
```

**Without phase conditioning:**
```python
(
    noisy_image,
    patch_coordinates,
    dataset_index,
    noisy_image_avg,
    noisy_image_std,
)  # 5-tuple
```

### Stage 4: Training Loop
**Files:** 
- `src/train.py` (lines 107-118, 179-182)
- `src/train_distributed.py` (lines 100-119, 139)

**Batch Unpacking:**
```python
if opt.is_zarr and opt.use_phase_conditioning:
    (img, _, ds_idx, mean, std, phase_sin, phase_cos) = batch
    phase_sin = phase_sin.cuda()
    phase_cos = phase_cos.cuda()
else:
    (img, _, ds_idx, mean, std) = batch
    phase_sin = None
    phase_cos = None
```

**Model Forward Call:**
```python
if use_phase_conditioning and phase_sin is not None:
    output = model(img, phase_sin, phase_cos)
else:
    output = model(img)
```

### Stage 5: Model Processing
**File:** `model/SUPPORT.py` (lines 509-604)

**Architecture Changes:**
1. **U-Net input channels:** 3×(T-1) instead of T-1
   - T-1 image channels + T-1 sin channels + T-1 cos channels
   - Phase broadcasted from (B, T-1, H) to (B, T-1, H, W)

2. **BS-Net first layer:** 3 channels instead of 1
   - 1 image channel (center) + sin + cos of center frame
   - Shape: (B, 3, H, W)

3. **Phase injection at deeper BS-Net layers:** Learned 1×1 projections
   - Input: (B, 3, H, W) phase-augmented tensor
   - Projects to blind_conv_channels for feature re-injection
   - Replaces scalar multiplication without phase

**Forward Pass Logic:**
```python
def forward(x, phase_sin=None, phase_cos=None):
    # x shape: (B, T, H, W)
    # phase_sin, phase_cos shape: (B, T, H)
    
    center = in_channels // 2
    
    # Split phase like image channels
    if use_phase_conditioning and phase_sin is not None:
        unet_phase_sin = cat([phase_sin[:,:center,:], phase_sin[:,center+1:,:]])
        unet_phase_cos = cat([phase_cos[:,:center,:], phase_cos[:,center+1:,:]])
        bsnet_phase_sin = phase_sin[:, center, :]   # (B, H)
        bsnet_phase_cos = phase_cos[:, center, :]
        
        # Broadcast U-Net phase to spatial dims
        u_sin = unet_phase_sin.unsqueeze(-1).expand(B, T-1, H, W)
        u_cos = unet_phase_cos.unsqueeze(-1).expand(B, T-1, H, W)
        
        # Concatenate with images
        unet_in = cat([images, u_sin, u_cos], dim=1)  # (B, 3*(T-1), H, W)
    
    # Process through U-Net and BS-Net with phase context
    # ...
    
    return output
```

---

## Key Implementation Details

### Phase Tensor Shapes

| Point in Pipeline | Shape | Interpretation |
|-------------------|-------|-----------------|
| Zarr position_signal | (T, H, W) | Raw sinusoid per frame, row, column |
| After extract_phase | (T, H) | Phase angle (converted to sin/cos) |
| In dataset per file | (T, H) | Stored in phase_sin_list/phase_cos_list |
| In batch | (B, T, H) | B batches of T-frame patches, H rows each |
| After split in forward | U-Net: (B, T-1, H), BS-Net: (B, H) | Separated by frame type |
| After broadcast in forward | (B, T-1, H, W) for U-Net | Expanded to spatial dimensions |
| After concatenation | (B, 3*(T-1), H, W) for U-Net | Combined with images and partner cos/sin |

### Model Architecture Adjustments

**When `use_phase_conditioning=True`:**

1. **UNet._gen_unet():**
   ```python
   unet_in_channels = 3 * (in_channels - 1)  # Instead of (in_channels - 1)
   ```

2. **BSNet._gen_bsnet() creates phase projection layers:**
   ```python
   self.inject_proj_3x3 = nn.ModuleList([
       nn.Conv2d(3, blind_conv_channels, kernel_size=1)
       for _ in range(depth3x3 - 1)
   ])
   self.inject_proj_5x5 = nn.ModuleList([
       nn.Conv2d(3, blind_conv_channels, kernel_size=1)
       for _ in range(depth5x5 - 1)
   ])
   ```

3. **First layer blind conv input channels:**
   ```python
   c_in = 3 if use_phase_conditioning else 1  # First layer
   # Subsequent layers always blind_conv_channels
   ```

4. **BS-Net first conv layers:**
   ```python
   bs_in_channels = one_by_one_channels[-1] + 2  # +2 for sin/cos
   ```

### Data Augmentation Impact

**Random transforms are conditionally applied:**
```python
if is_rotate and phase_sin is None:
    # 90-degree rotations ENABLED (no phase, rotation safe)
else:
    # 90-degree rotations DISABLED (phase is per-row, rotation unsafe)

if phase_sin is not None:
    # Flips applied to both images and phase along same axis
    phase_sin = torch.flip(phase_sin, dims=[2])  # Flip rows
    phase_cos = torch.flip(phase_cos, dims=[2])
```

**Rationale:** Phase encodes per-row information; 90° rotation would mix row/column dimensions, making phase meaningless.

### Distributed Training

**DDP Configuration:**
```python
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=opt.prevent_injection or opt.use_phase_conditioning,
)
```

**Why:** Phase projection layers may not be used in all paths (if `prevent_injection=True`). Setting `find_unused_parameters=True` prevents DDP errors about unused parameters.

---

## Critical Code Paths

### Path 1: Without Phase Conditioning
```
Batch: (img, coords, ds_idx, mean, std) - 5 elements
Model call: model(img)
U-Net: T-1 channels, no phase
BS-Net: 1 channel first layer, scalar injection
```

### Path 2: With Phase Conditioning (zarr + is_raw)
```
Batch: (img, coords, ds_idx, mean, std, phase_sin, phase_cos) - 7 elements
Model call: model(img, phase_sin, phase_cos)
U-Net: 3*(T-1) channels [img, sin, cos]
BS-Net: 3 channels first layer, 1x1 projection layers
```

### Path 3: With Phase Conditioning (zarr + not is_raw)
```
Phase extraction SKIPPED (only happens with is_raw=True)
Behaves like Path 1 (no phase)
```

---

## Discovered Issues

### 1. Missing Parameter in train.py
**File:** `src/train.py` lines 349-357

Current code does NOT pass `use_phase_conditioning` to `gen_train_dataloader()`:
```python
# MISSING PARAMETER
dataloader_train = gen_train_dataloader(
    opt.patch_size,
    opt.patch_interval,
    opt.batch_size,
    noisy_data,
    opt,
    is_zarr=opt.is_zarr,
    use_splatting=opt.use_splatting,
    # use_phase_conditioning=opt.use_phase_conditioning,  # MISSING!
)
```

**Impact:** Phase conditioning won't work with `train.py` main function (only works with `train_distributed.py`)

**Fix:** Add the parameter:
```python
dataloader_train = gen_train_dataloader(
    ...,
    use_phase_conditioning=opt.use_phase_conditioning,  # ADD THIS
)
```

### 2. Missing Parameter in train.py Model Initialization
**File:** `src/train.py` lines 388-401

Model is created without `use_phase_conditioning` parameter:
```python
model = SUPPORT(
    ...,
    # use_phase_conditioning=opt.use_phase_conditioning,  # MISSING!
)
```

**Impact:** Even if dataloader has phase, model won't use it (no 3×(T-1) channels, no projection layers)

**Fix:** Add the parameter:
```python
model = SUPPORT(
    ...,
    use_phase_conditioning=opt.use_phase_conditioning,
)
```

---

## Testing Checklist

- [ ] Flag is recognized: `python train.py --help | grep phase_conditioning`
- [ ] Phase extraction works: Verify zarr has `store["position"]`
- [ ] Batch format correct: Check dataloader returns 7-tuple with phase
- [ ] Model initialization correct: SUPPORT has `use_phase_conditioning=True`
- [ ] Model forward pass works: Can call `model(x, phase_sin, phase_cos)`
- [ ] Training runs without errors: Loss decreases over epochs
- [ ] DDP training works: No "unused parameters" errors with `find_unused_parameters=True`
- [ ] Phase improves performance: Validate loss/metrics better than baseline

---

## Summary of Files Modified/Involved

| File | Role | Status |
|------|------|--------|
| `src/utils/util.py` | Flag definition | Complete |
| `src/utils/dataset.py` | Data loading, phase extraction | Complete |
| `src/train.py` | Training loop | Incomplete (missing 2 parameters) |
| `src/train_distributed.py` | Distributed training loop | Complete |
| `model/SUPPORT.py` | Model architecture | Complete |

---

## Command Examples

### Correct Usage (train_distributed.py):
```bash
python src/train_distributed.py \
    --use_phase_conditioning \
    --is_zarr \
    --is_raw \
    --noisy_data /path/to/zarr/files \
    --batch_size 8 \
    --n_epochs 100
```

### Issue: Using train.py
```bash
python src/train.py \
    --use_phase_conditioning \
    --is_zarr \
    --is_raw \
    ...  # Phase won't be extracted or used due to missing parameters
```

---

## References

- Main trace document: `/gpfs/data/shohamlab/tom/support/PHASE_CONDITIONING_TRACE.md`
- Quick reference: `/gpfs/data/shohamlab/tom/support/PHASE_CONDITIONING_QUICK_REF.md`
- Test files: `tests/test_phase_conditioning.py`, `tests/test_phase_structure.py`

