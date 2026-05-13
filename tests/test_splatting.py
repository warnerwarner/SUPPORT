# %% Checking the learning rate of a model. It might be too slow


import torch
import numpy as np
import sys
import os
import json
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import defaultdict
import zarr
from torch.nn.parallel import DistributedDataParallel as DDP

# %%
# Add project root to path so we can import our modules
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
project_root = "/gpfs/data/shohamlab/tom/support"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.dataset import extract_phase
from model.SUPPORT import SUPPORT
from src.utils.dataset import gen_train_dataloader, random_transform
from src.utils.util import parse_arguments
from torch.utils.data import DataLoader

from src.utils.dataset import DatasetSUPPORT_test_stitch, extract_phase
from src.utils.alignment import calculate_peak_shifts
from src.utils.alignedzarr import AlignedZarr
from src.utils.splatting_loss import compute_splatting_loss


def train(
    train_dataloader,
    model,
    optimizer,
    scaler,
    rng,
    writer,
    epoch,
    opt,
    loss_option="l1_l2",
    loss_coef=[1.0, 1.0],
):
    """
    Train a model for a single epoch (non-distributed version)

    Arguments:
        train_dataloader: (Pytorch DataLoader)
        model: (Pytorch nn.Module)
        optimizer: (Pytorch optimizer)
        scaler: (GradScaler for AMP)
        rng: numpy random number generator
        writer: (Tensorboard writer)
        epoch: epoch of training (int)
        opt: argparse dictionary

    Returns:
        loss_list: list of total loss of each batch ([float])
        loss_list_l1: list of L1 loss of each batch ([float])
        loss_list_l2: list of L2 loss of each batch ([float])
    """

    is_rotate = True if model.bs_size[0] == model.bs_size[1] else False

    # initialize
    model.train()
    loss_list_l1 = []
    loss_list_l2 = []
    loss_list = []
    grads = defaultdict(list)

    if loss_option == "l1_l2":
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        loss_coef = opt.loss_coef
    elif loss_option == "huber":
        huber_pixelwise = torch.nn.HuberLoss(delta=0.2)

    if opt.use_splatting:
        epoch_factor = np.exp(-epoch / opt.reg_decay_epochs)

    dataloader_iter = tqdm(train_dataloader)

    # training
    for i, data in enumerate(dataloader_iter):
        patch_coords = None
        if opt.is_zarr and opt.use_phase_conditioning:
            (
                noisy_image,
                patch_coords,
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
            (noisy_image, patch_coords, ds_idx, noisy_image_avg, noisy_image_std) = data
            phase_sin = None
            phase_cos = None
            # For splatting: reshape to [B, 1, 1, 1, 1] to broadcast across [B, T, 2, H, W]
            # For normal: reshape to [B, 1, 1, 1] to broadcast across [B, T, H, W]
            if opt.use_splatting:
                noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1, 1))
                noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1, 1))
            else:
                noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1))
                noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1))
        else:
            (noisy_image, _, ds_idx) = data

        # Handle 4D (splatting) vs 4D/3D (normal zarr) data
        if opt.use_splatting:
            B, T, C, X, Y = noisy_image.shape  # [B, T, 2, H, W]
        else:
            shape_vals = noisy_image.shape
            if len(shape_vals) == 5:
                B, T, C, X, Y = shape_vals  # [B, T, C, H, W]
            else:
                B, T, X, Y = shape_vals  # [B, T, H, W]
                C = None

        noisy_image = noisy_image.cuda()

        # Random transforms - skip for splatting mode (rotation complicates position channel)
        if not opt.use_splatting:
            noisy_image, _, phase_sin, phase_cos = random_transform(
                noisy_image, None, rng, is_rotate, phase_sin, phase_cos
            )

        if opt.is_zarr:
            noisy_image_avg = noisy_image_avg.cuda()
            noisy_image_std = noisy_image_std.cuda()
            if opt.use_splatting:
                # Normalize only the EOD channel (channel 0) for splatting
                # noisy_image shape: [B, T, 2, H, W]
                # noisy_image_avg shape: [B, 1, 1, 1, 1]
                noisy_image[:, :, 0:1, :, :] = (
                    noisy_image[:, :, 0:1, :, :] - noisy_image_avg
                ) / noisy_image_std
            else:
                noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std

        # Prepare target for non-splatting mode only
        if not opt.use_splatting:
            noisy_image_target = torch.unsqueeze(
                noisy_image[:, int(T / 2), :, :], dim=1
            )

        optimizer.zero_grad()
        # Forward pass wrapped in autocast for AMP
        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            if phase_sin is not None:
                noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
            else:
                if opt.use_splatting and patch_coords is not None:
                    # patch_coords: [B,3,2], dims are [t,y,z] start/end
                    patch_origin = patch_coords[:, 1, 0:1]
                    patch_origin = torch.cat(
                        [patch_coords[:, 1, 0:1], patch_coords[:, 2, 0:1]], dim=1
                    ).to(noisy_image.device)
                    full_w = train_dataloader.dataset.noisy_images[0].shape[-1]
                    full_n_samples = torch.full(
                        (patch_origin.shape[0],),
                        int(full_w),
                        dtype=torch.int64,
                        device=noisy_image.device,
                    )
                    noisy_image_denoised = model(
                        noisy_image,
                        patch_origin=patch_origin,
                        full_n_samples=full_n_samples,
                    )
                else:
                    noisy_image_denoised = model(noisy_image)

            if opt.use_splatting:
                loss_dict = compute_splatting_loss(
                    model, noisy_image, noisy_image_denoised, epoch_factor, opt
                )
                loss_sum = loss_dict["total"]
            elif loss_option == "l1_l2":
                loss_l1_pixelwise = L1_pixelwise(
                    noisy_image_denoised, noisy_image_target
                )
                loss_l2_pixelwise = L2_pixelwise(
                    noisy_image_denoised, noisy_image_target
                )
                loss_sum = (
                    loss_coef[0] * loss_l1_pixelwise + loss_coef[1] * loss_l2_pixelwise
                )
            elif loss_option == "huber":
                loss_sum = huber_pixelwise(noisy_image_denoised, noisy_image_target)

        # Backward pass with GradScaler if AMP is enabled
        scaler.scale(loss_sum).backward()

        # Gradient clipping for splatting mode
        if (
            opt.use_splatting
            and hasattr(opt, "grad_clip_max")
            and opt.grad_clip_max > 0
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip_max)

        scaler.step(optimizer)
        scaler.update()

        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name].append(p.grad.norm().item())

        if not opt.use_splatting and loss_option == "l1_l2":
            loss_list_l1.append(loss_l1_pixelwise.item())
            loss_list_l2.append(loss_l2_pixelwise.item())
        loss_list.append(loss_sum.item())

        # print log
        if (epoch % opt.logging_interval == 0) and (
            i % opt.logging_interval_batch == 0
        ):
            loss_mean = np.mean(np.array(loss_list))
            if not opt.use_splatting:
                loss_mean_l1 = np.mean(np.array(loss_list_l1))
                loss_mean_l2 = np.mean(np.array(loss_list_l2))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # save model, optimizer, and scaler

    return loss_list, loss_list_l1, loss_list_l2, grads


# %% Old model
opt = parse_arguments(
    args=[
        "--training_size",
        "2",
        "--noisy_data",
        "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run012",
        "--patch_size",
        "61",
        "16",
        "128",
        "--patch_interval",
        "200",
        "8",
        "320",
        "--input_frames",
        "61",
        "--bs_size",
        "1",
        "2",
        "--depth",
        "5",
        "--is_zarr",
        "--is_raw",
        "--is_folder",
        "--use_splatting",  # Enable splatting in data loader
        "--splatting_params_json",
        "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run012/260430_splat_params.json",
        "--batch_size",
        "8",
    ]
)

# This will test the SplattingZarrWrapper
dataloader_train = gen_train_dataloader(
    opt.patch_size,
    opt.patch_interval,
    opt.batch_size,
    opt.noisy_data,
    opt,
    is_zarr=opt.is_zarr,
    is_raw=opt.is_raw,
    use_splatting=opt.use_splatting,  # Enable splatting in data loader
)

# Load splatting config if enabled
splatting_config = None
if opt.use_splatting:
    if not opt.splatting_params_json:
        raise ValueError(
            "--splatting_params_json must be provided when --use_splatting is enabled"
        )

    with open(opt.splatting_params_json, "r") as f:
        splatting_params = json.load(f)

    # Config for LearnableSplattingStage
    # n_lines and n_samples are inferred from input image dimensions
    splatting_config = {
        "initial_params": splatting_params,
        "drop_lines": 1,  # Preprocessing: drop 1 line from top
    }
    print(f"Loaded splatting parameters from {opt.splatting_params_json}")

model_base = SUPPORT(
    in_channels=opt.input_frames,
    mid_channels=opt.unet_channels,
    depth=opt.depth,
    blind_conv_channels=opt.blind_conv_channels,
    one_by_one_channels=opt.one_by_one_channels,
    last_layer_channels=opt.last_layer_channels,
    bs_size=opt.bs_size,
    bp=opt.bp,
    use_splatting=opt.use_splatting,
    splatting_config=splatting_config if opt.use_splatting else None,
    use_point_offset=True,
    point_offset_config={"hidden_channels": 8, "max_offset_px": 1.0},
).cuda()

optimizer = torch.optim.Adam(model_base.parameters(), lr=opt.lr)
scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
rng = np.random.default_rng(2231)  # Different seed per rank

outs_old = train(
    dataloader_train,
    model_base,
    optimizer,
    scaler,
    rng,
    None,
    0,
    opt,
    # loss_option="robust",
)

# %%
for d in dataloader_train:
    print(len(d))
    break
# %%
plt.imshow(model_base._last_splatted[4].mean(axis=0).cpu().detach().numpy())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True
from src.test_splatting import TemporalRawWindowDataset

zarr_in = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run011/stim_v1_fov3_440Hz_10X_2x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00002.zarr"
with zarr.open(zarr_in, mode="r") as store:
    eod_arr = store["eod"]
    pos_arr = store["position"]

    eod_mean = eod_arr.attrs.get("mean", None)
    eod_std = eod_arr.attrs.get("std", None)
    if eod_mean is None or eod_std is None:
        sample = eod_arr[:]
        eod_mean = float(sample.mean())
        eod_std = float(sample.std())

    dataset = TemporalRawWindowDataset(
        eod_arr=eod_arr,
        pos_arr=pos_arr,
        input_frames=61,
        eod_mean=eod_mean,
        eod_std=eod_std,
    )

loader = DataLoader(
    dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
)

outputs = []
center_indices = []
with torch.no_grad():
    for x, center_idx in tqdm(loader, desc="validate_splatting"):
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            y = model_base(x)  # [B, 1, H', W']
        outputs.append(y[:, 0].cpu().numpy())
        center_indices.append(center_idx.numpy())

denoised = np.concatenate(outputs, axis=0)  # [N_windows, H', W']
centers = np.concatenate(center_indices, axis=0)

order = np.argsort(centers)
denoised = denoised[order]
centers = centers[order]

# %%
plt.imshow(denoised.mean(axis=0), aspect="auto")

# %%
plt.imshow(loader.dataset.pos_arr[0], aspect="auto")
# %%
dir(model_base.splatting_stage)
for i in model_base.splatting_stage.get_params_dict():
    print(i, model_base.splatting_stage.get_params_dict()[i])


# %%
def infer_from_train_batch(model, batch, dataset, use_splatting=True, device="cuda"):
    """
    Run inference on a training-style dataloader batch with the same preprocessing
    used in train().

    Args:
        model: SUPPORT model
        batch: one item from dataloader_train
        dataset: dataloader_train.dataset (for full width lookup)
        use_splatting: whether model expects [B,T,2,H,W]
        device: "cuda" or "cpu"

    Returns:
        denoised: [B,1,H',W'] (or [B,1,H,W] in non-splatting mode)
        splatted: [B,T,H',W'] if available else None
        x_in: normalized model input tensor
    """
    model = model.to(device).eval()

    noisy_image = batch[0].to(device)
    patch_coords = batch[1] if len(batch) > 1 else None
    ds_idx = batch[2] if len(batch) > 2 else None
    noisy_mean = batch[3].to(device) if len(batch) > 3 else None
    noisy_std = batch[4].to(device) if len(batch) > 4 else None

    x_in = noisy_image.clone()
    if noisy_mean is not None and noisy_std is not None:
        if use_splatting:
            noisy_mean = noisy_mean.view(-1, 1, 1, 1, 1)
            noisy_std = noisy_std.view(-1, 1, 1, 1, 1)
            x_in[:, :, 0:1, :, :] = (x_in[:, :, 0:1, :, :] - noisy_mean) / (
                noisy_std + 1e-8
            )
        else:
            noisy_mean = noisy_mean.view(-1, 1, 1, 1)
            noisy_std = noisy_std.view(-1, 1, 1, 1)
            x_in = (x_in - noisy_mean) / (noisy_std + 1e-8)

    patch_origin = None
    full_n_samples = None
    if use_splatting and patch_coords is not None:
        patch_coords = patch_coords.to(device)
        patch_origin = torch.cat(
            [patch_coords[:, 1, 0:1], patch_coords[:, 2, 0:1]], dim=1
        )

        if ds_idx is not None:
            full_widths = []
            for i in range(ds_idx.shape[0]):
                dsi = int(ds_idx[i].item())
                full_widths.append(int(dataset.noisy_images[dsi].shape[-1]))
            full_n_samples = torch.tensor(full_widths, dtype=torch.int64, device=device)
        else:
            full_w = int(dataset.noisy_images[0].shape[-1])
            full_n_samples = torch.full(
                (x_in.shape[0],), full_w, dtype=torch.int64, device=device
            )

    with torch.no_grad():
        denoised = model(
            x_in,
            patch_origin=patch_origin,
            full_n_samples=full_n_samples,
        )
        splatted = getattr(model, "_last_splatted", None)

    return denoised, splatted, x_in


# %%
batch = next(iter(dataloader_train))
den, splat, x_in = infer_from_train_batch(
    model_base, batch, dataloader_train.dataset, use_splatting=True
)

# %%
plt.imshow(splat[0].mean(axis=0).cpu().detach().numpy())
plt.imshow(den[0, 0].cpu().detach().numpy())
