# Complete Flow Trace: --use_phase_conditioning Flag

## Overview
The `--use_phase_conditioning` flag enables the SUPPORT model to use sinusoidal phase information extracted from position signals in raw zarr data. Phase is represented as sin and cos values per row and frame, providing spatial phase context to the denoising model.

---

## 1. COMMAND-LINE ARGUMENT DEFINITION

### File: `/gpfs/data/shohamlab/tom/support/src/utils/util.py` (Lines 215-219)

```python
parser.add_argument(
    "--use_phase_conditioning",
    action="store_true",
    help="Use phase conditioning in the model",
)
```

**Details:**
- Flag name: `--use_phase_conditioning`
- Type: Boolean flag (`action="store_true"`)
- Default: `False` (flag not present = False, `--use_phase_conditioning` = True)
- Help text: "Use phase conditioning in the model"

**Accessed as:** `opt.use_phase_conditioning`

---

## 2. DATALOADER CREATION PATH

### File: `/gpfs/data/shohamlab/tom/support/src/train_distributed.py` (Lines 279-289)

```python
dataloader_train = gen_train_dataloader(
    opt.patch_size,
    opt.patch_interval,
    opt.batch_size,
    opt.noisy_data,
    opt,
    is_zarr=opt.is_zarr,
    is_raw=opt.is_raw,
    rank=rank,
    use_phase_conditioning=opt.use_phase_conditioning,  # PASSED HERE
)
```

### File: `/gpfs/data/shohamlab/tom/support/src/train.py` (Lines 349-357)

```python
dataloader_train = gen_train_dataloader(
    opt.patch_size,
    opt.patch_interval,
    opt.batch_size,
    noisy_data,
    opt,
    is_zarr=opt.is_zarr,
    use_splatting=opt.use_splatting,
    # NOTE: Missing use_phase_conditioning parameter in main function
)
```

---

## 3. PHASE EXTRACTION IN DATALOADER

### File: `/gpfs/data/shohamlab/tom/support/src/utils/dataset.py` (Lines 713-847)

#### Function Signature:
```python
def gen_train_dataloader(
    patch_size,
    patch_interval,
    batch_size,
    noisy_data_list,
    opt,
    is_zarr=False,
    is_raw=False,
    rank=0,
    use_phase_conditioning=False,  # PARAMETER HERE
    use_splatting=False,
):
```

#### Phase List Initialization (Lines 740-742):
```python
noisy_images_train = []
phase_sin_list = [] if use_phase_conditioning else None
phase_cos_list = [] if use_phase_conditioning else None
```

**Logic:** Lists are created only if `use_phase_conditioning=True`, otherwise they remain `None`.

#### Phase Extraction from Zarr (Lines 818-828):
```python
elif is_raw:
    noisy_image = store["eod"]
    
    if use_phase_conditioning:
        position_signal = store["position"][:]  # Load position data
        phase_sin, phase_cos = extract_phase(position_signal)
        if phase_sin.is_cuda:
            phase_sin = phase_sin.cpu()
            phase_cos = phase_cos.cpu()
        phase_sin_list.append(phase_sin)  # Per-file phase data
        phase_cos_list.append(phase_cos)
```

**Flow:**
1. Check if `use_phase_conditioning=True` AND `is_raw=True`
2. Load position signal from zarr: `store["position"][:]`
3. Call `extract_phase()` function (see below)
4. Ensure phase data is on CPU (move from GPU if needed)
5. Append to phase lists for each data file

#### Dataset Creation (Lines 838-847):
```python
dataset_train = DatasetSUPPORT(
    noisy_images_train,
    patch_size=patch_size,
    patch_interval=patch_interval,
    transform=None,
    random_patch=True,
    load_to_memory=not is_zarr,
    phase_sin_list=phase_sin_list if use_phase_conditioning else None,
    phase_cos_list=phase_cos_list if use_phase_conditioning else None,
)
```

**Key:** Phase lists are passed to DatasetSUPPORT only if flag is enabled.

---

## 4. PHASE EXTRACTION FUNCTIONS

### File: `/gpfs/data/shohamlab/tom/support/src/utils/dataset.py` (Lines 85-98)

```python
def extract_phase(position_signal):
    """
    Extract phase from sinusoidal position signal using Hilbert transform.
    Automatically chooses GPU or CPU implementation based on availability.

    Args:
        position_signal: (T, H, W) numpy array with sinusoids along W dimension
    Returns:
        phase_sin, phase_cos: (T, H) tensors - sin/cos of phase for continuity
    """
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        return _extract_phase_gpu(position_signal)
    else:
        return _extract_phase_cpu(position_signal)
```

**Output Shape:** `(T, H)` - phase value per frame and row

#### CPU Implementation (Lines 25-47):
```python
def _extract_phase_cpu(position_signal):
    # Input: (T, H, W)
    T, H, W = position_signal.shape
    
    # Compute Hilbert transform along W dimension (sinusoid axis)
    analytic = hilbert(position_signal, axis=2)
    
    # Extract phase at center pixel W//2
    phase = np.angle(analytic[:, :, W // 2])  # Shape: (T, H)
    
    # Convert to sin/cos for numerical stability
    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)
    
    return phase_sin, phase_cos
```

**Process:**
1. Compute analytic signal via Hilbert transform along width (W) dimension
2. Extract phase angle at center pixel: `phase = angle(analytic[:, :, W//2])`
3. Convert to sin/cos: `sin(phase)` and `cos(phase)`

#### GPU Implementation (Lines 50-82):
- Similar logic but uses `torch.fft` for GPU acceleration
- Splits data into batches for memory efficiency
- Uses `hilbert_torch()` function

---

## 5. DATASET BATCH GENERATION

### File: `/gpfs/data/shohamlab/tom/support/src/utils/dataset.py` (Lines 523-620)

#### DatasetSUPPORT.__getitem__ Method:

**When loading from zarr with phase conditioning (Lines 557-593):**

```python
if self.phase_sin_list is not None:
    # Phase has shape (T_full, H) — slice matching time and row dims
    phase_sin = self.phase_sin_list[ds_idx][t_range, y_range]
    phase_cos = self.phase_cos_list[ds_idx][t_range, y_range]
    phase_sin = torch.tensor(phase_sin, dtype=torch.float32)
    phase_cos = torch.tensor(phase_cos, dtype=torch.float32)
else:
    phase_sin = None
    phase_cos = None

# ... image loading ...

if self.phase_sin_list is not None:
    return (
        noisy_image,
        patch_coordinates,
        ds_idx,
        noisy_image_avg,
        noisy_image_std,
        phase_sin,      # Shape: (T_patch, H_patch)
        phase_cos,      # Shape: (T_patch, H_patch)
    )
else:
    return (
        noisy_image,
        patch_coordinates,
        ds_idx,
        noisy_image_avg,
        noisy_image_std,
    )
```

**Return Tuples:**

**With phase conditioning (7 elements):**
```
(
    noisy_image,          # (T, H, W)
    patch_coordinates,    # Spatial indices
    ds_idx,              # Dataset index
    noisy_image_avg,     # Mean for normalization
    noisy_image_std,     # Std for normalization
    phase_sin,           # (T_patch, H_patch)
    phase_cos,           # (T_patch, H_patch)
)
```

**Without phase conditioning (5 elements):**
```
(
    noisy_image,
    patch_coordinates,
    ds_idx,
    noisy_image_avg,
    noisy_image_std,
)
```

---

## 6. TRAINING DATA UNPACKING

### File: `/gpfs/data/shohamlab/tom/support/src/train.py` (Lines 107-118)

```python
# Handle different data formats based on configuration
if opt.is_zarr and getattr(opt, 'use_phase_conditioning', False):
    (
        noisy_image,
        _,
        ds_idx,
        noisy_image_avg,
        noisy_image_std,
        phase_sin,
        phase_cos,
    ) = data
    phase_sin = phase_sin.cuda()
    phase_cos = phase_cos.cuda()
```

**Flow:**
1. Check both `is_zarr` AND `use_phase_conditioning`
2. Unpack 7-tuple batch
3. Move phase tensors to CUDA device
4. Initialize phase_sin/cos = None for non-zarr paths

### File: `/gpfs/data/shohamlab/tom/support/src/train_distributed.py` (Lines 100-119)

```python
if opt.is_zarr and opt.use_phase_conditioning:
    (
        noisy_image,
        _,
        ds_idx,
        noisy_image_avg,
        noisy_image_std,
        phase_sin,
        phase_cos,
    ) = data
    phase_sin = phase_sin.cuda()
    phase_cos = phase_cos.cuda()
    noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1))
    noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1))
elif opt.is_zarr:
    (noisy_image, _, ds_idx, noisy_image_avg, noisy_image_std) = data
    # phase_sin/cos remain None
```

---

## 7. MODEL FORWARD PASS WITH PHASE

### File: `/gpfs/data/shohamlab/tom/support/src/train.py` (Lines 179-182)

```python
elif getattr(opt, 'use_phase_conditioning', False) and phase_sin is not None:
    noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
else:
    noisy_image_denoised = model(noisy_image)
```

### File: `/gpfs/data/shohamlab/tom/support/src/train_distributed.py` (Line 139)

```python
noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
```

**Note:** train_distributed.py always passes phase (may be None).

---

## 8. MODEL INITIALIZATION WITH FLAG

### File: `/gpfs/data/shohamlab/tom/support/src/train.py` (Lines 388-401)

```python
model = SUPPORT(
    in_channels=opt.input_frames,
    mid_channels=opt.unet_channels,
    depth=opt.depth,
    blind_conv_channels=opt.blind_conv_channels,
    one_by_one_channels=opt.one_by_one_channels,
    last_layer_channels=opt.last_layer_channels,
    bs_size=opt.bs_size,
    bp=opt.bp,
    use_splatting=opt.use_splatting,
    splatting_config=splatting_config,
    use_point_offset=opt.use_point_offset,
    point_offset_config=point_offset_config,
    # NOTE: Missing use_phase_conditioning parameter
)
```

### File: `/gpfs/data/shohamlab/tom/support/src/train_distributed.py` (Lines 349-361)

```python
model = SUPPORT(
    in_channels=opt.input_frames,
    mid_channels=opt.unet_channels,
    depth=opt.depth,
    blind_conv_channels=opt.blind_conv_channels,
    one_by_one_channels=opt.one_by_one_channels,
    last_layer_channels=opt.last_layer_channels,
    bs_size=opt.bs_size,
    bp=opt.bp,
    is_raw=opt.is_raw,
    prevent_injection=opt.prevent_injection,
    use_phase_conditioning=opt.use_phase_conditioning,  # PASSED HERE
)
```

---

## 9. SUPPORT MODEL ARCHITECTURE WITH PHASE CONDITIONING

### File: `/gpfs/data/shohamlab/tom/support/model/SUPPORT.py`

#### Constructor (Lines 18-35):

```python
def __init__(
    self,
    ...
    use_phase_conditioning=False,
    ...
):
    ...
    self.use_phase_conditioning = use_phase_conditioning
```

#### U-Net Channel Adjustment (Lines 147-153):

```python
def _gen_unet(self):
    # When phase conditioning is enabled, concatenate sin+cos phase channels
    # for each non-center frame: (T-1) image + (T-1) sin + (T-1) cos = 3*(T-1)
    unet_in_channels = self.in_channels - 1
    if self.use_phase_conditioning:
        unet_in_channels = 3 * (self.in_channels - 1)
```

**Logic:**
- Normal mode: `in_channels - 1` (exclude center frame)
- Phase mode: `3 * (in_channels - 1)` (image + sin + cos for T-1 frames)

Example: With 61 input frames:
- Normal: 60 channels
- Phase: 180 channels (60 images + 60 sin + 60 cos)

#### Blind Spot Net Phase Projection Layers (Lines 230-245):

```python
# Per-layer 1x1 convs to project phase-augmented input (3 ch) into
# blind_conv_channels for re-injection at layers c >= 1.
# Only created when phase conditioning is active.
if self.use_phase_conditioning:
    self.inject_proj_3x3 = nn.ModuleList(
        [
            nn.Conv2d(3, self.blind_conv_channels, kernel_size=1)
            for _ in range(self.depth3x3 - 1)
        ]
    )
    self.inject_proj_5x5 = nn.ModuleList(
        [
            nn.Conv2d(3, self.blind_conv_channels, kernel_size=1)
            for _ in range(self.depth5x5 - 1)
        ]
    )
```

**Purpose:** Project 3-channel phase-augmented input (center frame + sin + cos) to blind_conv_channels for feature re-injection.

#### BS-Net First Layer Channels (Lines 249-252):

```python
# Calculate input channels: UNet output + 2 (sin/cos of center-frame row phase)
bs_in_channels = self.one_by_one_channels[-1]
if self.use_phase_conditioning:
    bs_in_channels += 2  # Add sin and cos channels
```

**Logic:**
- Normal mode: UNet output channels only
- Phase mode: UNet output + 2 (phase sin and cos for center frame)

#### Blind Spot Conv Input Channels (Lines 288-325):

For both 3x3 and 5x5 blind convolutions:

```python
for d in range(self.depth3x3):
    if d == 0:
        c_in = 3 if self.use_phase_conditioning else 1
    else:
        c_in = self.blind_conv_channels
```

**Logic:**
- First layer: 3 channels with phase (center frame + sin + cos), else 1 (center frame only)
- Subsequent layers: blind_conv_channels (output of previous layers)

---

## 10. FORWARD PASS WITH PHASE

### File: `/gpfs/data/shohamlab/tom/support/model/SUPPORT.py` (Lines 509-604)

#### Main Forward Method Signature:
```python
def forward(self, x, phase_sin=None, phase_cos=None, patch_origin=None, full_n_samples=None):
```

#### Phase Splitting and Broadcasting (Lines 566-583):

```python
if self.use_phase_conditioning and phase_sin is not None:
    # phase_sin/cos: (B, T, H) -> split like the image channels
    # U-Net gets non-center frames: (B, T-1, H)
    unet_phase_sin = torch.cat(
        [phase_sin[:, :center, :], phase_sin[:, center + 1 :, :]], dim=1
    )
    unet_phase_cos = torch.cat(
        [phase_cos[:, :center, :], phase_cos[:, center + 1 :, :]], dim=1
    )
    # BS-net gets center frame only: (B, H)
    bsnet_phase_sin = phase_sin[:, center, :]
    bsnet_phase_cos = phase_cos[:, center, :]

    # Broadcast U-Net phase from (B, T-1, H) to (B, T-1, H, W) and concat
    B, T_minus_1, H, W = unet_in.shape
    u_sin = unet_phase_sin.unsqueeze(-1).expand(B, T_minus_1, H, W)
    u_cos = unet_phase_cos.unsqueeze(-1).expand(B, T_minus_1, H, W)
    unet_in = torch.cat([unet_in, u_sin, u_cos], dim=1)
```

**Flow:**
1. **Phase input shape:** (B, T, H) - phase per batch, frame, and row
2. **Extraction of center frame index:** `center = in_channels // 2`
3. **U-Net phase:** Takes non-center frames from phase_sin/cos
   - Shape after split: (B, T-1, H)
   - Broadcast to (B, T-1, H, W) by expanding W dimension
   - Concatenate to image channels: [images, sin, cos]
4. **BS-Net phase:** Takes only center frame phase
   - Shape: (B, H)

#### U-Net Input Construction:
```python
unet_in = torch.cat(
    [
        x[:, :center, :, :],      # Non-center images (B, T-1, H, W)
        x[:, center + 1 :, :, :],
    ],
    dim=1,
)
# Then phase is concatenated:
# unet_in = [images(B, T-1, H, W), u_sin(B, T-1, H, W), u_cos(B, T-1, H, W)]
# Final shape: (B, 3*(T-1), H, W)
```

#### BS-Net Input with Phase (Lines 403-450):

```python
def forward_bsnet(self, x, unet_out, phase_sin=None, phase_cos=None):
    # x : bsnet input (B, 1, H, W) — raw center frame
    
    # Build phase-augmented input for injection
    if self.use_phase_conditioning and phase_sin is not None:
        B, C, H, W = x.shape
        # Reshape phase from (B, H) to (B, 1, H, W)
        p_sin = phase_sin.unsqueeze(1).unsqueeze(-1).expand(B, 1, H, W)
        p_cos = phase_cos.unsqueeze(1).unsqueeze(-1).expand(B, 1, H, W)
        x_inject = torch.cat([x, p_sin, p_cos], dim=1)  # (B, 3, H, W)
    else:
        x_inject = x  # (B, 1, H, W)
    
    # Concatenate phase to U-Net output for first layers
    if unet_out is not None:
        if self.use_phase_conditioning and phase_sin is not None:
            B_u, C_u, H_u, W_u = unet_out.shape
            up_sin = phase_sin.unsqueeze(1).unsqueeze(-1).expand(B_u, 1, H_u, W_u)
            up_cos = phase_cos.unsqueeze(1).unsqueeze(-1).expand(B_u, 1, H_u, W_u)
            unet_out = torch.cat([unet_out, up_sin, up_cos], dim=1)
        
        # Process concatenated unet_out with conv layers
        unet_out1 = self.conv3x3[0](unet_out)
        unet_out1 = self.conv3x3[1](unet_out1)
        unet_out1 = self.conv3x3[2](unet_out1)
    
    # Main blind spot convolution loops
    for c in range(self.depth3x3):
        if c == 0:
            x1 = x_inject  # First layer uses phase-augmented input
        else:
            # Feature re-injection at deeper layers
            if not self.prevent_injection:
                if self.use_phase_conditioning and phase_sin is not None:
                    # Use learned 1x1 conv projection for phase features
                    x1 = x1 + self.inject_proj_3x3[c - 1](x_inject)
                else:
                    # Use original scalar injection with 1-channel input
                    x1 = x1 + (self.scalars_3x3[c - 1] * x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
```

**Phase Input at BS-Net Layers:**
- **First layer (c=0):** `x_inject = (B, 3, H, W)` = center frame + sin + cos
- **Deeper layers (c>=1):** Phase re-injected via `inject_proj_3x3[c-1]` projection layers
  - Input: `x_inject` (B, 3, H, W)
  - Projects to blind_conv_channels for feature addition

---

## 11. DATA FLOW DIAGRAM

```
COMMAND LINE
    |
    v
opt.use_phase_conditioning (boolean flag)
    |
    |---> gen_train_dataloader(use_phase_conditioning=opt.use_phase_conditioning)
    |         |
    |         v
    |     If use_phase_conditioning=True AND is_raw=True:
    |         |
    |         v
    |     extract_phase(position_signal)
    |         |
    |         v
    |     Hilbert transform -> phase -> sin/cos
    |         |
    |         v
    |     phase_sin_list, phase_cos_list created per file
    |         |
    |         v
    |     DatasetSUPPORT(phase_sin_list, phase_cos_list)
    |         |
    |         v
    |     Batch: (image, coords, ds_idx, mean, std, phase_sin, phase_cos)
    |
    |---> Training Loop (train.py / train_distributed.py)
    |         |
    |         v
    |     Unpack: phase_sin, phase_cos from batch (if use_phase_conditioning)
    |         |
    |         v
    |     model(image, phase_sin, phase_cos)
    |
    |---> SUPPORT Model (model/SUPPORT.py)
    |         |
    |         v
    |     self.use_phase_conditioning = use_phase_conditioning
    |         |
    |         v
    |     Architecture Modifications:
    |     - U-Net input: 3*(T-1) channels (image + sin + cos for T-1 frames)
    |     - BS-Net first layer: 3 channels (image + sin + cos for center frame)
    |     - Inject layers: 1x1 convs to project phase features
    |         |
    |         v
    |     forward(x, phase_sin, phase_cos):
    |         |
    |         v
    |     Split phase into U-Net and BS-Net portions
    |         |
    |         v
    |     U-Net: Concatenate [images, sin, cos] -> 3*(T-1) input
    |     BS-Net: Concatenate [image, sin, cos] -> 3 input at first layer
    |              Inject phase via learned projections at deeper layers
    |         |
    |         v
    |     Output: Denoised center frame
```

---

## 12. RANDOM TRANSFORM WITH PHASE

### File: `/gpfs/data/shohamlab/tom/support/src/utils/dataset.py` (Lines 116-164)

```python
def random_transform(
    input, target, rng, is_rotate=True, phase_sin=None, phase_cos=None
):
    # Disable rotations when phase conditioning is active
    # Phase is per-row and rotation would mix row/column dimensions
    if is_rotate and phase_sin is None:
        # Apply 90-degree rotations
        ...
    
    if rand_num_2 == 1:
        # Flip along dim 2 (row axis)
        input = torch.flip(input, dims=[2])
        if phase_sin is not None:
            # Flip phase along its row axis (dim 2) to match
            phase_sin = torch.flip(phase_sin, dims=[2])
            phase_cos = torch.flip(phase_cos, dims=[2])
```

**Key Behaviors:**
1. **Rotations disabled:** When `phase_sin is not None` (phase conditioning active)
   - Reason: Phase is per-row; 90° rotation would mix row/column dimensions
2. **Flips applied:** Phase tensors are flipped along same axis as images (dim 2 = row axis)

---

## 13. DISTRIBUTED TRAINING SPECIAL HANDLING

### File: `/gpfs/data/shohamlab/tom/support/src/train_distributed.py` (Line 374)

```python
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=opt.prevent_injection or opt.use_phase_conditioning,
)
```

**Purpose:** When `use_phase_conditioning=True`, some parameters (inject_proj layers) may not be used in all forward passes if prevent_injection=True. Setting `find_unused_parameters=True` prevents DDP errors.

---

## 14. SUMMARY TABLE

| Component | With Phase Conditioning | Without Phase Conditioning |
|-----------|------------------------|---------------------------|
| **Flag** | `--use_phase_conditioning` | (omitted) |
| **Phase Lists** | Created in gen_train_dataloader | None |
| **Extract Phase** | From position signal via Hilbert | N/A |
| **Batch Tuple** | 7-tuple with phase_sin/cos | 5-tuple |
| **U-Net Channels** | 3*(T-1) = [img, sin, cos] | T-1 = [img] |
| **U-Net Input** | (B, 3*(T-1), H, W) | (B, T-1, H, W) |
| **BS-Net First Layer** | 3 channels (img+sin+cos) | 1 channel (img) |
| **BS-Net Inject Layers** | 1x1 conv projections | Scalar multiplication |
| **Random Rotations** | Disabled | Enabled |
| **Forward Pass Call** | `model(x, phase_sin, phase_cos)` | `model(x)` |
| **Distributed DDP** | find_unused_parameters=True | depends on prevent_injection |

---

## 15. EXAMPLE EXECUTION PATHS

### Path 1: With Phase Conditioning Enabled

```
python train.py --use_phase_conditioning --is_zarr --is_raw

1. parse_arguments() -> opt.use_phase_conditioning = True
2. gen_train_dataloader(..., use_phase_conditioning=True)
   - Load zarr files
   - Extract position_signal from store["position"]
   - phase_sin, phase_cos = extract_phase(position_signal)
   - phase_sin_list = [phase_sin_file1, phase_sin_file2, ...]
   - phase_cos_list = [phase_cos_file1, phase_cos_file2, ...]
   - Create DatasetSUPPORT with phase lists
3. Training loop:
   - For each batch: (img, coords, ds_idx, mean, std, phase_sin, phase_cos)
   - phase_sin, phase_cos to CUDA
   - model(img, phase_sin, phase_cos)
4. Model forward:
   - phase_sin, phase_cos shape: (B, T, H)
   - Split into unet_phase and bsnet_phase
   - U-Net processes (B, 3*(T-1), H, W) = [images, sin, cos]
   - BS-Net processes (B, 3, H, W) at first layer
   - Inject phase via learned projections at deeper layers
   - Output: denoised center frame
```

### Path 2: Without Phase Conditioning

```
python train.py --is_zarr

1. parse_arguments() -> opt.use_phase_conditioning = False
2. gen_train_dataloader(..., use_phase_conditioning=False)
   - phase_sin_list = None
   - phase_cos_list = None
   - Create DatasetSUPPORT without phase lists
3. Training loop:
   - For each batch: (img, coords, ds_idx, mean, std)
   - No phase tensors
   - model(img)
4. Model forward:
   - phase_sin = None, phase_cos = None
   - U-Net processes (B, T-1, H, W) = [images only]
   - BS-Net processes (B, 1, H, W) at first layer
   - Inject features via scalar multiplication at deeper layers
   - Output: denoised center frame
```

---

## Key Architectural Changes Summary

### When `use_phase_conditioning=True`:

1. **Data Loading:**
   - Extract phase from zarr position signal
   - Phase shape: (T, H) per file
   - Convert to sin/cos for numerical stability

2. **U-Net:**
   - Input channels: 3*(T-1) instead of T-1
   - First conv layer expects concatenated [images, phase_sin, phase_cos]

3. **BS-Net:**
   - First layer input: 3 channels [center_image, phase_sin, phase_cos]
   - Deeper layers: Phase injected via learned 1x1 projection layers
   - No scalar injection when phase is present

4. **Training:**
   - Rotations disabled (phase is per-row)
   - Flips applied to both images and phase
   - DDP parameter handling for distributed training

5. **Information Flow:**
   - Phase provides spatial context (per-row sinusoidal information)
   - U-Net uses phase context for all non-center frames
   - BS-Net uses phase context for center frame processing
   - Learned projections combine phase with network features

