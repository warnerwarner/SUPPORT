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

opt_align = parse_arguments(
    args=[
        "--training_size",
        "2",
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
        "--bs_size",
        "1",
        "3",
        "--depth",
        "5",
        "--is_zarr",
        "--is_raw",
        "--is_folder",
        "--align_data",
    ]
)
dataloader_train = gen_train_dataloader(
    opt_align.patch_size,
    opt_align.patch_interval,
    opt_align.batch_size,
    opt_align.noisy_data,  # Pass file paths, not pre-loaded tensors
    opt_align,
    is_zarr=opt_align.is_zarr,
    is_raw=opt_align.is_raw,
    rank=0,  # Pass rank for distributed cache synchronization
    use_phase_conditioning=False,
)
# %%
for d in dataloader_train.dataset.noisy_images:
    print(d.shape)
    break

# %%
plt.imshow(dataloader_train.dataset.noisy_images[0][1, :, :200])

# %%
print(type(dataloader_train.dataset.noisy_images[0]))
# %%


sys.path.append("/gpfs/home/warnet02/data/")

import eo_accelerated_2p as eo2p
import importlib

importlib.reload(eo2p.reconstruction)

recon_file_aligned = "/gpfs/home/warnet02/data/stephen/run012/default_recon_params.json"
# %%
d = dataloader_train.dataset.noisy_images[0][:]
p = dataloader_train.dataset.noisy_images[0]._test_get(slice(None))
# %%
dp = np.zeros((d.shape[0] * 2, d.shape[1], d.shape[2]))
dp[0::2] = d
dp[1::2] = p
# %%


binned_norm = eo2p.reconstruction.reconstruct(
    dp[:500, :], recon_file_aligned, save_tiff=False, return_binned=True
)
# %%
plt.imshow(binned_norm.mean(axis=0))
# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

im = ax[0].imshow(dp[:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(dp[1:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im, ax=ax[1])
im = ax[2].imshow(binned_norm.mean(axis=0), aspect="auto")
plt.colorbar(im, ax=ax[2])
plt.suptitle("Average of 250 frames")
ax[0].set_title("Raw")
ax[1].set_title("EOD")
ax[2].set_title("Reconstructed")

# %%
with zarr.open(opt_align.noisy_data[0], mode="r") as z:
    raw_d = z["eod"][:]
    raw_p = z["position"][:]


# %%
raw_dp = np.zeros((raw_d.shape[0] * 2, raw_d.shape[1], raw_d.shape[2]))
raw_dp[0::2] = raw_d
raw_dp[1::2] = raw_p
# %%
binned_norm_raw = eo2p.reconstruction.reconstruct(
    raw_dp[:500, :], recon_file_aligned, save_tiff=False, return_binned=True
)
# %%
binned_norm_raw.shape
# %%
plt.imshow(binned_norm_raw.mean(axis=0))
# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im = ax[0].imshow(raw_dp[:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(raw_dp[1:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im, ax=ax[1])
im = ax[2].imshow(binned_norm_raw.mean(axis=0), aspect="auto")
plt.colorbar(im, ax=ax[2])
# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
im1 = ax[0, 0].imshow(dp[:500:2, :, :200].mean(axis=0), aspect="auto")

plt.colorbar(im1, ax=ax[0, 0])
im2 = ax[0, 1].imshow(dp[1:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im2, ax=ax[0, 1])
im3 = ax[0, 2].imshow(binned_norm.mean(axis=0), aspect="auto")
plt.colorbar(im3, ax=ax[0, 2])
plt.suptitle("Average of 250 frames")
ax[0, 0].set_title("Raw")
ax[0, 1].set_title("EOD")
ax[0, 2].set_title("Reconstructed")
im4 = ax[1, 0].imshow(raw_dp[:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im4, ax=ax[1, 0])
im5 = ax[1, 1].imshow(raw_dp[1:500:2, :, :200].mean(axis=0), aspect="auto")
plt.colorbar(im5, ax=ax[1, 1])
im6 = ax[1, 2].imshow(binned_norm_raw.mean(axis=0), aspect="auto")
plt.colorbar(im6, ax=ax[1, 2])
plt.suptitle("Average of 250 frames")
vmax1 = max(im1.get_array().max(), im4.get_array().max())
vmin1 = min(im1.get_array().min(), im4.get_array().min())
vmax2 = max(im2.get_array().max(), im5.get_array().max())
vmin2 = min(im2.get_array().min(), im5.get_array().min())
vmax3 = max(im3.get_array().max(), im6.get_array().max())
vmin3 = min(im3.get_array().min(), im6.get_array().min())
im1.set_clim(vmin1, vmax1)
im4.set_clim(vmin1, vmax1)
im2.set_clim(vmin2, vmax2)
im5.set_clim(vmin2, vmax2)
im3.set_clim(vmin3, vmax3)
im6.set_clim(vmin3, vmax3)
ax[1, 0].set_title("Raw")
ax[1, 1].set_title("EOD")
ax[1, 2].set_title("Reconstructed")

# %%
fig, ax = plt.subplots(3, 2, figsize=(12, 9))
im1 = ax[0, 0].imshow(binned_norm.mean(axis=0), aspect="auto")
plt.colorbar(im1, ax=ax[0, 0])
im2 = ax[1, 0].imshow(binned_norm_raw.mean(axis=0), aspect="auto")
plt.colorbar(im2, ax=ax[1, 0])
vmin = min(im1.get_array().min(), im2.get_array().min())
vmax = max(im1.get_array().max(), im2.get_array().max())
im1.set_clim(vmin, vmax)
im2.set_clim(vmin, vmax)
im3 = ax[2, 0].imshow(
    binned_norm_raw.mean(axis=0) - binned_norm.mean(axis=0), aspect="auto", cmap="bwr"
)
vmax = np.max(np.abs(im3.get_array()))
im3.set_clim(-vmax, vmax)
plt.colorbar(im3, ax=ax[2, 0])

ax[0, 0].set_title("Aligned 250 frames")
ax[1, 0].set_title("Raw 250 frames")
ax[2, 0].set_title("Raw - Aligned 250 frames")
im4 = ax[0, 1].imshow(binned_norm[0], aspect="auto")
plt.colorbar(im4, ax=ax[0, 1])
im5 = ax[1, 1].imshow(binned_norm_raw[0], aspect="auto")
plt.colorbar(im5, ax=ax[1, 1])
vmin = min(im4.get_array().min(), im5.get_array().min())
vmax = max(im4.get_array().max(), im5.get_array().max())
im4.set_clim(vmin, vmax)
im5.set_clim(vmin, vmax)
im6 = ax[2, 1].imshow(binned_norm_raw[0] - binned_norm[0], aspect="auto", cmap="bwr")
vmax = np.max(np.abs(im6.get_array()))
im6.set_clim(-vmax, vmax)
plt.colorbar(im6, ax=ax[2, 1])
ax[0, 1].set_title("Aligned 1 frame")
ax[1, 1].set_title("Raw 1 frame")
ax[2, 1].set_title("Raw - Aligned 1 frame")
plt.tight_layout()

# %%
opt_align = parse_arguments(
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
        "50",
        "8",
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
        "--random_seed",
        "4421",
        "--align_data",
    ]
)
dataloader_train = gen_train_dataloader(
    opt_align.patch_size,
    opt_align.patch_interval,
    opt_align.batch_size,
    opt_align.noisy_data,  # Pass file paths, not pre-loaded tensors
    opt_align,
    is_zarr=opt_align.is_zarr,
    is_raw=opt_align.is_raw,
    rank=0,  # Pass rank for distributed cache synchronization
    use_phase_conditioning=False,
)
model_aligned = SUPPORT(
    in_channels=opt_align.input_frames,
    mid_channels=opt_align.unet_channels,
    depth=opt_align.depth,
    blind_conv_channels=opt_align.blind_conv_channels,
    one_by_one_channels=opt_align.one_by_one_channels,
    last_layer_channels=opt_align.last_layer_channels,
    bs_size=opt_align.bs_size,
    bp=opt_align.bp,
).cuda()

optimizer = torch.optim.Adam(model_aligned.parameters(), lr=opt_align.lr)
scaler = torch.cuda.amp.GradScaler(enabled=opt_align.use_amp)
rng = np.random.default_rng(4421)  # Different seed per rank

outs = []
for epoch in range(5):
    # dataloader_train.dataset.precompute_indicies()
    outs_aligned = train(
        dataloader_train,
        model_aligned,
        optimizer,
        scaler,
        rng,
        None,
        epoch,
        opt_align,
    )
    print(f"Epoch {epoch} aligned loss: {np.mean(outs_aligned[0][-10:])}")
    outs.append(outs_aligned)
# %%
offset = 0
for i in outs:
    xs = np.arange(len(i[0])) + offset
    plt.plot(xs, i[0])
# %% test validation
test_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00005.zarr"
with zarr.open(test_path, mode="r") as store:
    test_noisy_d = store["eod"]
    test_noisy_pos = store["position"][:]
    shifts = calculate_peak_shifts(test_noisy_pos)
noisy_image = AlignedZarr(test_path, shifts=shifts)
ni = noisy_image[:]
# %%
patch_size = [61, 16, 320]
patch_interval = [1, 4, 160]
ni_torch = torch.from_numpy(ni.astype(np.float32)).type(torch.FloatTensor)
test_dataset = DatasetSUPPORT_test_stitch(
    ni_torch,
    patch_size=patch_size,
    patch_interval=patch_interval,
    phase_sin=None,
    phase_cos=None,
)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
# %%


def validate(test_dataloader, model):
    """
    Validate a model with a test data

    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(
            test_dataloader.dataset.noisy_image.shape, dtype=np.float32
        )

        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, (noisy_image, _, single_coordinate, phase_sin, phase_cos) in enumerate(
            tqdm(test_dataloader, desc="validate")
        ):
            noisy_image = noisy_image.cuda()  # [b, z, y, x]
            phase_sin = phase_sin.cuda() if phase_sin is not None else None
            phase_cos = phase_cos.cuda() if phase_cos is not None else None
            noisy_image_denoised = model(noisy_image, phase_sin, phase_cos).cpu()
            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)):
                stack_start_w = int(single_coordinate["stack_start_w"][bi])
                stack_end_w = int(single_coordinate["stack_end_w"][bi])
                patch_start_w = int(single_coordinate["patch_start_w"][bi])
                patch_end_w = int(single_coordinate["patch_end_w"][bi])

                stack_start_h = int(single_coordinate["stack_start_h"][bi])
                stack_end_h = int(single_coordinate["stack_end_h"][bi])
                patch_start_h = int(single_coordinate["patch_start_h"][bi])
                patch_end_h = int(single_coordinate["patch_end_h"][bi])

                stack_start_s = int(single_coordinate["init_s"][bi])

                denoised_stack[
                    stack_start_s + (T // 2),
                    stack_start_h:stack_end_h,
                    stack_start_w:stack_end_w,
                ] = noisy_image_denoised[bi].squeeze()[
                    patch_start_h:patch_end_h, patch_start_w:patch_end_w
                ]

        # change nan values to 0 and denormalize
        denoised_stack = (
            denoised_stack * test_dataloader.dataset.std_image.numpy()
            + test_dataloader.dataset.mean_image.numpy()
        )

        return denoised_stack


validated_stack = validate(testloader, model_aligned)
# %%
pos = noisy_image._test_get(slice(None))

# %%
vs_pos = np.zeros((pos.shape[0] * 2, pos.shape[1], pos.shape[2]))
vs_pos[0::2] = validated_stack
vs_pos[1::2] = pos
# %%

binned_norm_raw = eo2p.reconstruction.reconstruct(
    vs_pos, recon_file_aligned, save_tiff=False, return_binned=True
)
# %%
d = noisy_image[:]
# %%
dp = np.zeros((d.shape[0] * 2, d.shape[1], d.shape[2]))
dp[0::2] = d
dp[1::2] = pos
# %%


binned_norm = eo2p.reconstruction.reconstruct(
    dp, recon_file_aligned, save_tiff=False, return_binned=True
)
# %%
print(binned_norm_raw.shape)
# %%
plt.imshow(binned_norm_raw[30:].mean(axis=0), vmax=350)
plt.colorbar()
# %%
plt.imshow(binned_norm[30:].mean(axis=0), vmax=350)
plt.colorbar()
# %%
vmin = 80
vmax = 350
fig, ax = plt.subplots(2, 2, figsize=(15, 6))
im1 = ax[0, 0].imshow(binned_norm_raw[40], vmin=vmin, vmax=vmax)
im2 = ax[0, 1].imshow(binned_norm[40], vmin=vmin, vmax=vmax)
im3 = ax[1, 0].imshow(binned_norm_raw[30:80].mean(axis=0), vmin=vmin, vmax=vmax)
im4 = ax[1, 1].imshow(binned_norm[30:80].mean(axis=0), vmin=vmin, vmax=vmax)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar_ax.set_xticks([])
# switch y axis to other side
cbar_ax.yaxis.tick_right()
fig.colorbar(im3, cax=cbar_ax)
ax[0, 0].set_title("Denoised 1 frame")
ax[0, 1].set_title("Raw 1 frame")
ax[1, 0].set_title("Denoised 50 frames")
ax[1, 1].set_title("Raw 50 frames")
[i.set_axis_off() for i in ax.flatten()]

# %% making a comparison tiff
print(binned_norm_raw.shape, binned_norm.shape)
# %%
joined_binned = np.zeros(
    (binned_norm_raw.shape[0], binned_norm_raw.shape[1], binned_norm_raw.shape[2] * 2),
    dtype=np.float32,
)
joined_binned[:, :, : binned_norm_raw.shape[2]] = binned_norm_raw
joined_binned[:, :, binned_norm_raw.shape[2] :] = binned_norm
# %%
plt.imshow(joined_binned[35])
# %%
outpath = (
    "/gpfs/data/shohamlab/tom/stephen/comparison_tiffs/run012/comparison_0005.tiff"
)
import os
import tifffile

if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))
tifffile.imwrite(outpath, joined_binned.astype(np.float32), metadata={"axes": "TYX"})
