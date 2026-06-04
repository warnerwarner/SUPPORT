# %% Checking the learning rate of a model. It might be too slow


import torch
import numpy as np
import sys
import os
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


# %%
def robust_loss(x, alpha=1.0, c=1.0):
    x = x.float()
    eps = 1e-6
    a = abs(alpha - 2)
    return (a / (alpha + eps)) * (((x / c) ** 2 / a + 1.0) ** (alpha / 2) - 1)


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

    dataloader_iter = tqdm(train_dataloader)

    # training
    for i, data in enumerate(dataloader_iter):
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
            noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1))
            noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1))
            phase_sin = None
            phase_cos = None
        else:
            (noisy_image, _, ds_idx) = data

        B, T, X, Y = noisy_image.shape
        noisy_image = noisy_image.cuda()
        noisy_image, _, phase_sin, phase_cos = random_transform(
            noisy_image, None, rng, is_rotate, phase_sin, phase_cos
        )
        if opt.is_zarr:
            noisy_image_avg = noisy_image_avg.cuda()
            noisy_image_std = noisy_image_std.cuda()
            noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std
        noisy_image_target = torch.unsqueeze(noisy_image[:, int(T / 2), :, :], dim=1)

        optimizer.zero_grad()
        # Forward pass wrapped in autocast for AMP
        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            if phase_sin is not None:
                noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
            else:
                noisy_image_denoised = model(noisy_image)
            if loss_option == "l1_l2":
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
            elif loss_option == "robust":
                loss_sum = robust_loss(
                    noisy_image_denoised - noisy_image_target,
                    alpha=loss_coef[0],
                    c=loss_coef[1],
                ).sum()

        # Backward pass with GradScaler if AMP is enabled
        scaler.scale(loss_sum).backward()
        scaler.step(optimizer)
        scaler.update()

        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name].append(p.grad.norm().item())

        if loss_option == "l1_l2":
            loss_list_l1.append(loss_l1_pixelwise.item())
            loss_list_l2.append(loss_l2_pixelwise.item())
        loss_list.append(loss_sum.item())

        # print log
        if (epoch % opt.logging_interval == 0) and (
            i % opt.logging_interval_batch == 0
        ):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # save model, optimizer, and scaler

    return loss_list, loss_list_l1, loss_list_l2, grads


# %%
opt_old = parse_arguments(
    args=[
        "--training_size",
        "10",
        "--noisy_data",
        "/gpfs/data/shohamlab/tom/stephen/zarr/12-5-25-force1s-vis-stim/run011",
        "--patch_size",
        "61",
        "16",
        "320",
        "--patch_interval",
        "400",
        "16",
        "160",
        "--input_frames",
        "61",
        "--bs_size",
        "1",
        "3",
        "--depth",
        "5",
        "--is_zarr",
        "--is_raw",
        "--is_folder",
    ]
)
dataloader_train = gen_train_dataloader(
    opt_old.patch_size,
    opt_old.patch_interval,
    opt_old.batch_size,
    opt_old.noisy_data,  # Pass file paths, not pre-loaded tensors
    opt_old,
    is_zarr=opt_old.is_zarr,
    is_raw=opt_old.is_raw,
    rank=0,  # Pass rank for distributed cache synchronization
    use_phase_conditioning=False,
)
model_base = SUPPORT(
    in_channels=opt_old.input_frames,
    mid_channels=opt_old.unet_channels,
    depth=opt_old.depth,
    blind_conv_channels=opt_old.blind_conv_channels,
    one_by_one_channels=opt_old.one_by_one_channels,
    last_layer_channels=opt_old.last_layer_channels,
    bs_size=opt_old.bs_size,
    bp=opt_old.bp,
).cuda()

optimizer = torch.optim.Adam(model_base.parameters(), lr=opt_old.lr)
scaler = torch.cuda.amp.GradScaler(enabled=opt_old.use_amp)
rng = np.random.default_rng(2231)  # Different seed per rank

outs_old = train(
    dataloader_train,
    model_base,
    optimizer,
    scaler,
    rng,
    None,
    0,
    opt_old,
    # loss_option="robust",
)

outs_old1 = train(
    dataloader_train,
    model_base,
    optimizer,
    scaler,
    rng,
    None,
    1,
    opt_old,
    # loss_option="robust",
)

# %%
sys.path.append("/gpfs/data/shohamlab/tom/eo_accelerated_2p")
import splatting

# %%
import json

splat_params = json.load(
    open(
        "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/260430_splat_params.json",
        "r",
    )
)
# %%
for key, value in splat_params.items():
    print(f"{key}:")
    for kkey, vvalue in value.items():
        print(f"  {kkey}: {vvalue}")


# %%
def get_splat_args(
    series: np.ndarray,  # Read-only view on 3D image series (time, channel, j_slow, i_fast)
    eod_n_spots: int,  # Number of resolvable spots in the EOD scan. Determines final grid size
    eod_amplitude: float,  # Amplitude of the EOD scan relative to the slow axis. Set negative to reverse polarity
    eod_phase_rad: tuple[
        float, float
    ],  # Phase offset of the EOD recording to space in radians. Accounts for electronic delay, different for even/odd lines
    reso_phase_shift_px: tuple[
        float, float
    ],  # Phase shift in pixels to apply to the resonant galvo scan model. Accounts for even/odd line phase
    reso_temporal_fill_fraction: float,  # Fraction of the resonant galvo period that is recorded by ScanImage
    scan_aspect: float,  # Aspect ratio of the scan (fast/slow) as set by ScanImage
    eod_intensity_correction,  # Normalizes by the EOD scan intensity profile to correct for nonuniform illumination
    splat_grid_upsample_factor: int,  # Oversampling factor for the splat grid
    splat_sigma_px: float,  # Sigma of Gaussian kernel in pixels / splat_grid_upsample_factor
    impute_missing: bool = False,  # Whether to impute missing pixels in the final splatted image with noise
    verbose: bool = False,
):

    n_lines = series.shape[1]  # Lines per frame
    n_line = series.shape[2]  # Samples per line

    # Define splat grid size
    n_y = int(
        n_lines * eod_n_spots * splat_grid_upsample_factor
    )  # Oversample resolvable spots
    n_x = int(n_y * scan_aspect)

    n_modeled_line = int(
        n_line / reso_temporal_fill_fraction
    )  # Samples per modeled line. Includes blanked samples
    n_blanked = n_modeled_line - n_line
    n_total = int(n_lines * n_modeled_line)

    spatial_fill_fraction = np.sin(reso_temporal_fill_fraction * np.pi / 2)

    t = np.arange(n_total)  # Sample clock, including blanked samples
    y = np.linspace(0, n_y, n_total)

    # Resonant trajectory in normalized units
    x = np.cos(np.pi / n_modeled_line * t) / spatial_fill_fraction
    # Normalize to span 0..n_x (blanked samples fall outside and are clipped later)
    x = (x + 1.0) / 2.0 * n_x
    eod_amplitude_px = (
        eod_amplitude * n_y
    )  # Convert EOD amplitude from relative units to pixel units

    # Precompute scan model
    odd_mask = (np.arange(n_line) + n_blanked / 2 - reso_phase_shift_px[0]).astype(int)
    even_mask = (np.arange(n_line) + n_blanked / 2 + reso_phase_shift_px[1]).astype(int)
    scan_model = np.zeros((n_lines, n_line, 2), dtype=np.float32)
    for i in range(n_lines):
        if i % 2 == 0:  # Even line
            scan_model[i, :, 0] = y[even_mask + i * n_modeled_line]
            scan_model[i, :, 1] = x[even_mask + i * n_modeled_line]
        else:  # Odd line
            scan_model[i, :, 0] = y[odd_mask + i * n_modeled_line]
            scan_model[i, :, 1] = x[odd_mask + i * n_modeled_line]

    splat_args = {
        "scan_model": scan_model,
        "eod_amplitude_px": eod_amplitude_px,
        "eod_phase_rad": eod_phase_rad,
        "splat_sigma_px": splat_sigma_px,
        "splat_grid_upsample_factor": splat_grid_upsample_factor,
        "impute_missing": bool(impute_missing),
    }
    return splat_args, n_y, n_x


# %%
series = dataloader_train.dataset[0][0].numpy()[None, :, :]
splat_args, n_y, n_x = get_splat_args(
    series,
    eod_n_spots=splat_params["splatting"]["eod_n_spots"],
    eod_amplitude=splat_params["splatting"]["eod_amplitude"],
    eod_phase_rad=splat_params["splatting"]["eod_phase_rad"],
    reso_phase_shift_px=splat_params["splatting"]["reso_phase_shift_px"],
    reso_temporal_fill_fraction=splat_params["splatting"][
        "reso_temporal_fill_fraction"
    ],
    scan_aspect=splat_params["splatting"]["scan_aspect"],
    eod_intensity_correction=splat_params["splatting"]["eod_intensity_correction"],
    splat_grid_upsample_factor=splat_params["splatting"]["splat_grid_upsample_factor"],
    splat_sigma_px=splat_params["splatting"]["splat_sigma_px"],
)

print(n_y, n_x)
print(splat_args["scan_model"].shape)
for i in splat_args:
    print(i)

# %%
len(dataloader_train.dataset)
dir(dataloader_train.dataset.noisy_images[0].store)


# %%
dst = np.zeros((n_y, n_x), dtype=np.float32)
splatting._splat_frame(
    series,
    dst,
    splat_args["scan_model"],
    splat_args["eod_amplitude_px"],
    splat_args["eod_phase_rad"],
    splat_args["splat_sigma_px"],
    splat_args["splat_grid_upsample_factor"],
    impute_missing=False,
)
