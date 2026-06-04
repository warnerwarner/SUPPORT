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

opt = parse_arguments(
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
        "50",
        "8",
        "160",
        "--use_phase_conditioning",
        "--is_folder",
        "--is_raw",
        "--is_zarr",
        "--blind_conv_channels",
        "96",
        "--unet_channels",
        "96",
        "192",
        "384",
        "768",
        "1536",
        "--one_by_one_channels",
        "48",
        "24",
        "--last_layer_channels",
        "96",
        "48",
        "24",
        "--checkpoint_interval",
        "10",
        "--lr",
        "1e-3",
        "--batch_size",
        "32",
    ]
)
# %%
print(opt)
# %%


dataloader_train = gen_train_dataloader(
    opt.patch_size,
    opt.patch_interval,
    opt.batch_size,
    opt.noisy_data,  # Pass file paths, not pre-loaded tensors
    opt,
    is_zarr=opt.is_zarr,
    is_raw=opt.is_raw,
    rank=0,  # Pass rank for distributed cache synchronization
    use_phase_conditioning=opt.use_phase_conditioning,
)
# %%
dataloader = DataLoader(
    dataloader_train.dataset,
    batch_size=opt.batch_size,
    num_workers=7,
    prefetch_factor=opt.prefetch_factor,
)
# %%
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
    use_phase_conditioning=opt.use_phase_conditioning,
).cuda()

# %%

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
# Initialize GradScaler if AMP is enabled
scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
rng = np.random.default_rng(2231)  # Different seed per rank


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
outs = train(
    dataloader_train,
    model,
    optimizer,
    scaler,
    rng,
    None,
    0,
    opt,
)
# %%
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.abs().mean().item())


# %%
def plot_grads(grad_dict, grouping, offset=0):
    vals = []
    layers = []
    wbs = []

    for name in grad_dict.keys():
        if grouping in name:
            vals.append(np.array(grad_dict[name]))
            split_name = name.split(".")
            layers.append(split_name[1])
            if len(split_name) == 3:
                wbs.append(split_name[2])

    vals = np.array(vals)
    if len(wbs) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        # group vals by wb
        vals1 = vals[np.array(wbs) == "weight"]
        vals2 = vals[np.array(wbs) == "bias"]
        labels = np.array(layers)[np.array(wbs) == "weight"]
        im = ax[0].imshow(
            vals1[:, offset:],
            aspect="auto",
            interpolation="none",
            extent=[offset, vals1.shape[1], vals1.shape[0] - 0.5, -0.5],
        )
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title(grouping + " weight")
        ax[0].set_yticks(ticks=np.arange(len(labels)), labels=labels)
        im = ax[1].imshow(
            vals2[:, offset:],
            aspect="auto",
            interpolation="none",
            extent=[offset, vals2.shape[1], vals2.shape[0] - 0.5, -0.5],
        )
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title(grouping + " bias")
        ax[1].set_yticks(ticks=np.arange(len(labels)), labels=labels)
        # update the x ticks using offset
    else:
        fig = plt.figure(figsize=(8, 5))
        plt.imshow(vals[:, offset:], aspect="auto", interpolation="none")
        plt.colorbar()
        plt.yticks(ticks=np.arange(len(layers)), labels=layers)

    fig.suptitle(grouping)
    fig.supylabel("Layer")
    fig.supxlabel("Batch")


# %% Old model
opt_old = parse_arguments(
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
def run_model(opt):

    dataloader_train = gen_train_dataloader(
        opt.patch_size,
        opt.patch_interval,
        opt.batch_size,
        opt.noisy_data,  # Pass file paths, not pre-loaded tensors
        opt,
        is_zarr=opt.is_zarr,
        is_raw=opt.is_raw,
        rank=0,  # Pass rank for distributed cache synchronization
        use_phase_conditioning=False,
    )
    model_base = SUPPORT(
        in_channels=opt.input_frames,
        mid_channels=opt.unet_channels,
        depth=opt.depth,
        blind_conv_channels=opt.blind_conv_channels,
        one_by_one_channels=opt.one_by_one_channels,
        last_layer_channels=opt.last_layer_channels,
        bs_size=opt.bs_size,
        bp=opt.bp,
    ).cuda()

    optimizer = torch.optim.Adam(model_base.parameters(), lr=opt.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
    rng = np.random.default_rng()  # Different seed per rank

    outs = train(
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

    outs2 = train(
        dataloader_train,
        model_base,
        optimizer,
        scaler,
        rng,
        None,
        1,
        opt,
        # loss_option="robust",
    )
    return outs, outs2


# %%
for name, p in model_base.named_parameters():
    if p.grad is not None:
        print(name, p.grad.abs().mean().item())
# %%
plt.plot(outs_old[0], label="epoch 1")
plt.plot(
    np.arange(len(outs_old[0]), len(outs_old1[0]) + len(outs_old[0])),
    outs_old1[0],
    label="epoch 2",
)
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.title("Base model loss")
# %%
beg_names = list(set([name.split(".")[0] for name in outs_old[-1].keys()]))
for name in beg_names:
    plot_grads(outs_old[-1], name, offset=0)

# %%
grads = np.array([outs_old[-1][name] for name in outs_old[-1] if "enc_layers" in name])
plt.imshow(grads[:, :100], aspect="auto", vmax=0.5, interpolation="none")
plt.colorbar()


# %%
beg_names = list(set([name.split(".")[0] for name in outs_old[-1].keys()]))
for name in beg_names:
    plot_grads(outs_old[-1], name, offset=0)
    plot_grads(outs_old[-1], name, offset=300)

# %%

outs_old2 = train(
    dataloader_train,
    model_base,
    optimizer,
    scaler,
    rng,
    None,
    1,
    opt_old,
    loss_option="robust",
)
# %%
beg_names = list(set([name.split(".")[0] for name in outs_old2[-1].keys()]))
for name in beg_names:
    plot_grads(outs_old2[-1], name, offset=0)
    plot_grads(outs_old2[-1], name, offset=300)
# %%
plt.plot(outs_old[0][:], label="epoch 1")
plt.plot(
    np.arange(len(outs_old[0]), len(outs_old2[0]) + len(outs_old[0])),
    outs_old2[0],
    label="epoch 2",
)
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.legend()

# %% trying model with alignment
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
rng = np.random.default_rng(2231)  # Different seed per rank

outs_aligned = train(
    dataloader_train,
    model_aligned,
    optimizer,
    scaler,
    rng,
    None,
    0,
    opt_align,
)

# %%
a = (
    dataloader_train.dataset.noisy_images[0]
    ._test_get((2, slice(None), slice(0, 100)))
    .squeeze()
)
# %%
plt.imshow(a)
# %%

with zarr.open(opt_align.noisy_data[4], mode="r") as store:
    print(store["eod"].shape)
# %%
plt.plot(outs_aligned[0], label="aligned")
# plt.plot(outs_old[0], label="l1_l2")
# %%

outs_aligned2 = train(
    dataloader_train,
    model_aligned,
    optimizer,
    scaler,
    rng,
    None,
    1,
    opt_align,
)
# %%

plt.plot(outs_aligned[0], label="aligned")
plt.plot(
    np.arange(len(outs_aligned[0]), len(outs_aligned2[0]) + len(outs_aligned[0])),
    outs_aligned2[0],
    label="aligned epoch 2",
)
# %%

outs_aligned3 = train(
    dataloader_train,
    model_aligned,
    optimizer,
    scaler,
    rng,
    None,
    2,
    opt_align,
)
# %%
beg_names = list(set([name.split(".")[0] for name in outs_aligned3[-1].keys()]))
for name in beg_names:
    plot_grads(outs_aligned3[-1], name, offset=0)
# %%

plt.plot(outs_aligned[0], label="aligned epoch 1")
plt.plot(
    np.arange(len(outs_aligned[0]), len(outs_aligned2[0]) + len(outs_aligned[0])),
    outs_aligned2[0],
    label="aligned epoch 2",
)
plt.plot(
    np.arange(
        len(outs_aligned[0]) + len(outs_aligned2[0]),
        len(outs_aligned[0]) + len(outs_aligned2[0]) + len(outs_aligned3[0]),
    ),
    outs_aligned3[0],
    label="aligned epoch 3",
)
plt.legend()

# %%
# outs = []
for epoch in range(15, 20):
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
outs_huber = []
for epoch in range(5):
    outs_aligned = train(
        dataloader_train,
        model_aligned,
        optimizer,
        scaler,
        rng,
        None,
        epoch,
        opt_align,
        loss_option="huber",
    )
    print(f"Epoch {epoch} aligned loss: {np.mean(outs_aligned[0][-10:])}")
    outs_huber.append(outs_aligned)

# %%

x_offset = 0
for epoch in range(5):
    xs = np.arange(x_offset, x_offset + len(outs_huber[epoch][0]))
    plt.plot(xs, outs_huber[epoch][0], label=f"aligned epoch {epoch+1}")
    x_offset += len(outs_huber[epoch][0])
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Huber loss")
# %%

data_file = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00011.zarr"
demo_tiff = zarr.open(data_file, mode="r")["eod"][:]
demo_pos = zarr.open(data_file, mode="r")["position"][:]
shifts = calculate_peak_shifts(demo_pos)
aligned_zarr = AlignedZarr(data_file, shifts)
print("Loading aligned data...")
aligned_data = aligned_zarr[:]
print("Loading aligned pos...")
aligned_pos = aligned_zarr._test_get(slice(None))

demo_tif = torch.from_numpy(aligned_data.astype(np.float32)).type(torch.FloatTensor)


frame_size = 61
bs_size = [1, 3]
ds_size = 1
patch_size = [frame_size, 16, 320]
patch_interval = [1, 4, 160]
batch_size = 32  # lower it if memory exceeds.
testset = DatasetSUPPORT_test_stitch(
    demo_tif,
    patch_size=patch_size,
    patch_interval=patch_interval,
    phase_sin=None,
    phase_cos=None,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
)
phase_denoised = validate(testloader, model_aligned)
# %%

sys.path.append("/gpfs/home/warnet02/data/")

import eo_accelerated_2p as eo2p
import importlib

importlib.reload(eo2p.reconstruction)

p_stack = combine_data_and_pos(aligned_data, aligned_pos, model_aligned.in_channels)
# %%
import json

recon_file_aligned = "/gpfs/home/warnet02/data/stephen/run012/aligned_params.json"
with open(recon_file_aligned, "r") as f:
    recon_params = json.load(f)
# %%
binned = eo2p.reconstruction.reconstruct(
    p_stack[:100], recon_file_aligned, save_tiff=False, return_binned=True
)
xlim = slice(200, 300)
ylim = slice(0, 50)
plt.imshow(binned.mean(axis=0)[ylim, xlim], vmax=500, vmin=0)
# %%

recon_file = "/gpfs/home/warnet02/data/stephen/run012/default_recon_params.json"
n_stack = combine_data_and_pos(demo_tiff, demo_pos, model_aligned.in_channels)
binned_norm = eo2p.reconstruction.reconstruct(
    n_stack[:200], recon_file, save_tiff=False, return_binned=True
)
# %%
import scipy


def objective(x, data, control, param_base):
    params = param_base.copy()
    params["binning"]["eod_dphase_px"] = [int(round(x[0])), int(round(x[1]))]
    params["binning"]["upsample_factor"] = int(round(x[2]))
    params["binning"]["reso_fill_fraction"] = x[3]
    params["binning"]["parallelization_mode"] = None
    params["post_processing"]["eod_artifact_spectral_iso_sigma"] = x[4]
    recon = eo2p.reconstruction.reconstruct(
        data, params, save_tiff=False, return_binned=True
    )
    # Compare recon to control using some metric, e.g. mean squared error
    metric = np.mean((recon - control) ** 2)
    return metric


res = scipy.optimize.minimize(
    objective,
    x0=[0, 0, 1, 1, 1],
    args=(p_stack[:200], binned_norm, recon_params),
    bounds=[(-10, 10), (-10, 10), (1, 10), (0.1, 5), (0.1, 5)],
    method="Nelder-Mead",
)

# %%
vmax = 500
dvmax = 100
xlim = slice(200, 300)
ylim = slice(0, 50)
fig, ax = plt.subplots(3, 1, figsize=(5, 7))
im = ax[0].imshow(binned.mean(axis=0)[ylim, xlim], vmax=vmax, vmin=0)
plt.colorbar(im, ax=ax[0])
ax[0].set_title("Aligned recon")
im = ax[1].imshow(binned_norm.mean(axis=0)[ylim, xlim], vmax=vmax, vmin=0)
plt.colorbar(im, ax=ax[1])
ax[1].set_title("Unaligned recon")
im = ax[2].imshow(
    (binned - binned_norm).mean(axis=0)[ylim, xlim], vmax=dvmax, vmin=-dvmax, cmap="bwr"
)
plt.colorbar(im, ax=ax[2])
ax[2].set_title("Difference (aligned - unaligned)")
plt.tight_layout()
# %%
x_offset = 0
for epoch in range(20):
    xs = np.arange(x_offset, x_offset + len(outs[epoch][0]))
    plt.plot(xs, outs[epoch][0], label=f"aligned epoch {epoch+1}")
    x_offset += len(outs[epoch][0])
plt.xlabel("Batch")
plt.ylabel("Loss")
# %%
outs_huber = train(
    dataloader_train,
    model,
    optimizer,
    scaler,
    rng,
    None,
    0,
    opt,
    loss_option="huber",
)
# %%
outs_robust = train(
    dataloader_train,
    model,
    optimizer,
    scaler,
    rng,
    None,
    0,
    opt,
    loss_option="robust",
)
# %%
plt.plot(outs_robust[0], label="robust")
# %%
plt.plot(outs[0][25:], label="l1_l2")
# plt.plot(outs_huber[0], label='huber')

# %%
alpha_vals = [0.5, 0.8, 1.0]
c_vals = [0.1, 0.5, 1.0]
# outs = {}
# models = {}
for alpha in alpha_vals:
    for c in c_vals:

        # model = SUPPORT(
        #     in_channels=opt.input_frames,
        #     mid_channels=opt.unet_channels,
        #     depth=opt.depth,
        #     blind_conv_channels=opt.blind_conv_channels,
        #     one_by_one_channels=opt.one_by_one_channels,
        #     last_layer_channels=opt.last_layer_channels,
        #     bs_size=opt.bs_size,
        #     bp=opt.bp,
        #     is_raw=opt.is_raw,
        #     prevent_injection=opt.prevent_injection,
        #     use_phase_conditioning=opt.use_phase_conditioning,
        # ).cuda()
        model = models[(alpha, c)]
        dataloader_train = gen_train_dataloader(
            opt.patch_size,
            opt.patch_interval,
            opt.batch_size,
            opt.noisy_data,  # Pass file paths, not pre-loaded tensors
            opt,
            is_zarr=opt.is_zarr,
            is_raw=opt.is_raw,
            rank=0,  # Pass rank for distributed cache synchronization
            use_phase_conditioning=opt.use_phase_conditioning,
        )
        dataloader = DataLoader(
            dataloader_train.dataset,
            batch_size=opt.batch_size,
            num_workers=7,
            prefetch_factor=opt.prefetch_factor,
        )
        outs_robust = train(
            dataloader_train,
            model,
            optimizer,
            scaler,
            rng,
            None,
            0,
            opt,
            loss_option="robust",
            loss_coef=[alpha, c],
        )
        plt.plot(outs_robust[0], label=f"robust alpha={alpha} c={c}")
        outs[(alpha, c)] = [outs[(alpha, c)], outs_robust[0]]
