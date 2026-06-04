import torch
import numpy as np
import sys
import os
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter


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
        "10",
        "4",
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
# %%


def train(
    train_dataloader,
    model,
    optimizer,
    scaler,
    rng,
    writer,
    epoch,
    opt,
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

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    loss_coef = opt.loss_coef

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
            noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
            loss_l1_pixelwise = L1_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_l2_pixelwise = L2_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_sum = (
                loss_coef[0] * loss_l1_pixelwise + loss_coef[1] * loss_l2_pixelwise
            )

        # Backward pass with GradScaler if AMP is enabled
        scaler.scale(loss_sum).backward()
        scaler.step(optimizer)
        scaler.update()

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

    return loss_list, loss_list_l1, loss_list_l2


# %%
rng = np.random.default_rng(opt.random_seed)  # Different seed per rank
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
for i in dataloader.dataset:
    for j in i:
        print(j.shape)
    break
