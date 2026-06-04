# Quick Reference: --use_phase_conditioning Flag

## File Locations Summary

| File | Line(s) | Component |
|------|---------|-----------|
| `src/utils/util.py` | 215-219 | Flag definition |
| `src/utils/dataset.py` | 85-98 | `extract_phase()` function |
| `src/utils/dataset.py` | 25-47 | CPU phase extraction (Hilbert transform) |
| `src/utils/dataset.py` | 713-847 | `gen_train_dataloader()` function |
| `src/utils/dataset.py` | 740-742 | Phase list initialization |
| `src/utils/dataset.py` | 818-828 | Phase extraction from zarr |
| `src/utils/dataset.py` | 523-620 | `DatasetSUPPORT.__getitem__()` |
| `src/utils/dataset.py` | 557-593 | Batch return with phase |
| `src/utils/dataset.py` | 116-164 | `random_transform()` with phase |
| `src/train.py` | 107-118 | Batch unpacking (train.py) |
| `src/train.py` | 179-182 | Model forward call (train.py) |
| `src/train.py` | 349-357 | Dataloader creation (train.py) |
| `src/train_distributed.py` | 100-119 | Batch unpacking (train_distributed.py) |
| `src/train_distributed.py` | 139 | Model forward call (train_distributed.py) |
| `src/train_distributed.py` | 279-289 | Dataloader creation (train_distributed.py) |
| `src/train_distributed.py` | 374 | DDP find_unused_parameters |
| `model/SUPPORT.py` | 18-35 | Constructor |
| `model/SUPPORT.py` | 147-153 | U-Net channel adjustment |
| `model/SUPPORT.py` | 230-245 | Phase projection layers |
| `model/SUPPORT.py` | 249-252 | BS-Net first layer channels |
| `model/SUPPORT.py` | 288-325 | Blind conv input channels |
| `model/SUPPORT.py` | 403-450 | `forward_bsnet()` with phase |
| `model/SUPPORT.py` | 509-604 | Main `forward()` method |

## Data Flow Overview

```
opt.use_phase_conditioning (boolean)
         ↓
gen_train_dataloader(use_phase_conditioning=...)
         ↓
extract_phase(position_signal) if use_phase_conditioning
         ↓
phase_sin_list, phase_cos_list: List[(T, H) tensors]
         ↓
DatasetSUPPORT(phase_sin_list, phase_cos_list)
         ↓
Batch: (image, coords, ds_idx, mean, std, phase_sin, phase_cos)
         ↓
Training loop: model(image, phase_sin, phase_cos)
         ↓
SUPPORT.forward(x, phase_sin, phase_cos):
  - Split phase: U-Net & BS-Net portions
  - U-Net input: 3*(T-1) channels [img, sin, cos]
  - BS-Net: 3 channels at L0, inject via 1x1 convs at L>0
         ↓
Output: Denoised center frame
```

## Key Shapes Throughout Pipeline

| Stage | Shape | Notes |
|-------|-------|-------|
| **Position Signal (zarr)** | (T, H, W) | Sinusoid along W dimension |
| **Extracted Phase** | (T, H) | Per-frame, per-row phase values |
| **Phase (after reshape)** | (T, H) | Stored in dataset |
| **Batch Phase** | (B, T, H) | Batch of T-frame patches with H rows |
| **U-Net Phase** | (B, T-1, H, W) | Broadcast to spatial dims |
| **BS-Net Phase** | (B, H) | Center frame only |
| **U-Net Input** | (B, 3*(T-1), H, W) | [img, sin, cos] concatenated |
| **BS-Net Input L0** | (B, 3, H, W) | [img, sin, cos] concatenated |
| **Injection Input** | (B, 3, H, W) | x_inject for phase re-injection |

## Conditional Logic

### Data Unpacking (training loop)
```python
if opt.is_zarr and opt.use_phase_conditioning:
    # Unpack 7-tuple WITH phase
else:
    # Unpack 5-tuple WITHOUT phase
```

### Model Forward Call
```python
if use_phase_conditioning and phase_sin is not None:
    model(x, phase_sin, phase_cos)
else:
    model(x)
```

### Architecture Changes
```python
if use_phase_conditioning:
    # U-Net: 3*(T-1) channels
    # BS-Net: 3 channels first layer, projection layers for deeper
else:
    # U-Net: T-1 channels
    # BS-Net: 1 channel, scalar injection at deeper layers
```

### Data Augmentation
```python
if is_rotate and phase_sin is None:
    # Allow 90° rotations (phase_sin = None means no phase)
else:
    # Disable rotations (prevents mixing row/column dims when phase present)

if phase_sin is not None:
    # Apply flips to phase along same axis as images
```

## Critical Dependencies

1. **zarr + is_raw requirement:**
   - Phase extraction only happens when `is_zarr=True` AND `is_raw=True`
   - Position signal must be available in zarr: `store["position"]`

2. **Model initialization:**
   - **train_distributed.py:** Passes `use_phase_conditioning` to SUPPORT
   - **train.py:** Missing parameter in main function (BUG?)

3. **DDP training:**
   - When `use_phase_conditioning=True`, set `find_unused_parameters=True`
   - Prevents errors when phase projection layers not used in some paths

## Common Issues & Debugging

### Issue 1: No phase data extracted
- **Cause:** Missing `--use_phase_conditioning` flag
- **Fix:** Add flag when running training

### Issue 2: Shape mismatch in model
- **Cause:** Model initialized without `use_phase_conditioning=True`
- **Fix:** Ensure flag passed to SUPPORT constructor

### Issue 3: DDP errors about unused parameters
- **Cause:** `find_unused_parameters=False` with phase conditioning
- **Fix:** Set `find_unused_parameters=opt.use_phase_conditioning`

### Issue 4: Phase not applied to center frame in BS-Net
- **Cause:** Phase split logic correct - center frame has (B, H) shape
- **Fix:** Expected behavior - only center frame gets phase directly

## Testing Phase Conditioning

### Verify phase extraction:
```python
from src.utils.dataset import extract_phase
position_signal = zarr.open(...)["position"][:]  # Shape: (T, H, W)
phase_sin, phase_cos = extract_phase(position_signal)
print(phase_sin.shape)  # Should be (T, H)
print(phase_cos.shape)  # Should be (T, H)
```

### Verify batch format:
```python
from src.utils.dataset import gen_train_dataloader
dataloader = gen_train_dataloader(..., use_phase_conditioning=True)
batch = next(iter(dataloader))
print(len(batch))  # Should be 7 (with phase) or 5 (without)
if len(batch) == 7:
    img, coords, ds_idx, mean, std, p_sin, p_cos = batch
    print(f"Phase sin shape: {p_sin.shape}")  # (B, T, H)
```

### Verify model accepts phase:
```python
model = SUPPORT(..., use_phase_conditioning=True)
x = torch.randn(B, T, H, W)
p_sin = torch.randn(B, T, H)
p_cos = torch.randn(B, T, H)
output = model(x, p_sin, p_cos)  # Should work
```

