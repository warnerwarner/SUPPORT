import torch
import numpy as np
import sys
import zarr

sys.path.insert(0, "/gpfs/data/shohamlab/tom/support")

from model.SUPPORT import SUPPORT

# Load one zarr file directly
zarr_path = "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run011/stim_v1_fov3_440Hz_10X_2x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00001.zarr"
print(f"Loading {zarr_path}...")
data = zarr.open(zarr_path, mode="r")
print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys())}")
data = data["eod"]
print(f"Data shape: {data.shape}")

# Extract a small patch
T, H, W = data.shape
t_center = T // 2
h_start = H // 2 - 8
w_start = W // 2 - 160

noisy_image = data[
    t_center - 30 : t_center + 31, h_start : h_start + 16, w_start : w_start + 320
]
print(f"Patch shape: {noisy_image.shape}")

# Normalize
noisy_image_avg = noisy_image.mean()
noisy_image_std = noisy_image.std()
noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std

# Convert to tensor
noisy_image = torch.from_numpy(noisy_image).float().unsqueeze(0).cuda()
print(f"Tensor shape: {noisy_image.shape}")

# Generate fake phase data (B, T, H)
B, T, H, W = noisy_image.shape
phase_sin = torch.sin(
    torch.linspace(0, 2 * np.pi, T).unsqueeze(0).unsqueeze(2).expand(B, T, H)
).cuda()
phase_cos = torch.cos(
    torch.linspace(0, 2 * np.pi, T).unsqueeze(0).unsqueeze(2).expand(B, T, H)
).cuda()
print(f"Phase sin shape: {phase_sin.shape}")

# Create model
print("Creating model...")
model = SUPPORT(
    in_channels=61,
    mid_channels=[16, 32, 64],
    depth=3,
    blind_conv_channels=16,
    one_by_one_channels=[16, 8],
    last_layer_channels=[8, 4],
    bs_size=[2, 2],
    bp=False,
    is_raw=True,
    use_phase_conditioning=True,
)
model = model.cuda()
model.train()

print("Running forward pass...")
try:
    output = model(noisy_image, phase_sin, phase_cos)
    print(f"Output shape: {output.shape}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
