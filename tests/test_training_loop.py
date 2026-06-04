import torch
import numpy as np
import sys
import zarr

sys.path.insert(0, "/gpfs/data/shohamlab/tom/support")

from model.SUPPORT import SUPPORT

# Load 3 zarr files
zarr_paths = [
    "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run011/stim_v1_fov3_440Hz_10X_2x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00001.zarr",
    "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00001.zarr",
    "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run013/stim_v1_fov5_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00001.zarr",
]

datasets = []
for path in zarr_paths:
    print(f"Loading {path}...")
    data = zarr.open(path, mode="r")["eod"]
    print(f"  Shape: {data.shape}")
    datasets.append(data)

# Create model
print("Creating model...")
model = SUPPORT(
    in_channels=61,
    mid_channels=[16, 32, 64, 128, 256],
    depth=5,
    blind_conv_channels=64,
    one_by_one_channels=[32, 16],
    last_layer_channels=[64, 32, 16],
    bs_size=[2, 2],
    bp=False,
    is_raw=True,
    use_phase_conditioning=True,
)
model = model.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Extract random patches and run training loop
print("Running 5 training loops...")
for i in range(5):
    # Pick random file and random patch
    ds_idx = np.random.randint(len(datasets))
    ds = datasets[ds_idx]
    T, H, W = ds.shape

    t_start = np.random.randint(0, max(1, T - 61))
    h_start = np.random.randint(0, max(1, H - 16))
    w_start = np.random.randint(0, max(1, W - 320))

    noisy_image = ds[
        t_start : t_start + 61, h_start : h_start + 16, w_start : w_start + 320
    ]

    # Normalize
    noisy_image_avg = noisy_image.mean()
    noisy_image_std = noisy_image.std()
    noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std

    # Convert to tensor
    noisy_image = torch.from_numpy(noisy_image).float().unsqueeze(0).cuda()

    # Generate fake phase data
    B, T, H, W = noisy_image.shape
    phase_sin = torch.sin(
        torch.linspace(0, 2 * np.pi, T).unsqueeze(0).unsqueeze(2).expand(B, T, H)
    ).cuda()
    phase_cos = torch.cos(
        torch.linspace(0, 2 * np.pi, T).unsqueeze(0).unsqueeze(2).expand(B, T, H)
    ).cuda()

    # Target is middle frame
    noisy_image_target = noisy_image[:, T // 2, :, :].unsqueeze(1)

    optimizer.zero_grad()
    output = model(noisy_image, phase_sin, phase_cos)
    loss = criterion(output, noisy_image_target)
    loss.backward()
    optimizer.step()

    print(f"Loop {i + 1}/5: loss = {loss.item():.6f}")

print("SUCCESS!")
