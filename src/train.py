import os
import random
import logging
import time
import json
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.dataset import gen_train_dataloader, random_transform
from src.utils.util import parse_arguments
from model.SUPPORT import SUPPORT
from src.utils.splatting_loss import compute_splatting_loss
from src.utils.splatting_validation import (
    validate_splatting_params,
    visualize_splatting_output,
)
from collections import defaultdict


def robust_loss(x, alpha=1.0, c=1.0):
    """Robust loss function for outlier-resistant training"""
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
    loss_coef=None,
    track_gradients=False,
):
    """
    Train a model for a single epoch

    Arguments:
        train_dataloader: (Pytorch DataLoader)
        model: (Pytorch nn.Module)
        optimizer: (Pytorch optimizer)
        scaler: (GradScaler for AMP)
        rng: numpy random number generator
        writer: (Tensorboard writer)
        epoch: epoch of training (int)
        opt: argparse dictionary
        loss_option: type of loss to use - "l1_l2", "huber", or "robust" (default: "l1_l2")
        loss_coef: loss coefficients (default: uses opt.loss_coef or [1.0, 1.0])
        track_gradients: whether to track gradient norms (default: False)

    Returns:
        loss_list: list of total loss of each batch ([float])
        loss_list_l1: list of L1 loss of each batch ([float])
        loss_list_l2: list of L2 loss of each batch ([float])
        grads: gradient norms dict if track_gradients (optional)
    """
    if loss_coef is None:
        loss_coef = getattr(opt, "loss_coef", [1.0, 1.0])

    is_rotate = True if model.bs_size[0] == model.bs_size[1] else False

    # initialize
    model.train()
    loss_list_l1 = []
    loss_list_l2 = []
    loss_list = []
    grads = defaultdict(list) if track_gradients else {}

    # Initialize loss functions based on option
    if loss_option == "l1_l2":
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
    elif loss_option == "huber":
        huber_pixelwise = torch.nn.HuberLoss(delta=0.2)

    # training
    for i, data in enumerate(tqdm(train_dataloader)):
        patch_coords = None
        phase_sin = None
        phase_cos = None

        # Handle different data formats based on configuration
        if opt.is_zarr and getattr(opt, "use_phase_conditioning", False):
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
            (noisy_image, patch_coords, ds_idx, noisy_image_avg, noisy_image_std) = data
            noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1))
            noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1))
        else:
            (noisy_image, _, ds_idx) = data

        B, T, X, Y = noisy_image.shape  # [B, T, H, W]

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
            if getattr(opt, "use_phase_conditioning", False) and phase_sin is not None:
                noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
            else:
                noisy_image_denoised = model(noisy_image)

            # Handle different loss options
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

            if loss_option == "l1_l2":
                loss_list_l1.append(loss_l1_pixelwise.item())
                loss_list_l2.append(loss_l2_pixelwise.item())

        # Backward pass with GradScaler if AMP is enabled
        scaler.scale(loss_sum).backward()

        # Track gradients if requested
        if track_gradients:
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grads[name].append(p.grad.norm().item())

        scaler.step(optimizer)
        scaler.update()

        loss_list.append(loss_sum.item())

        # print log
        if (epoch % opt.logging_interval == 0) and (
            i % opt.logging_interval_batch == 0
        ):
            loss_mean = np.mean(np.array(loss_list))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if writer is not None:
                writer.add_scalar(
                    "Loss/train_batch", loss_mean, epoch * len(train_dataloader) + i
                )

            # Standard logging
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))
            if writer is not None:
                writer.add_scalar(
                    "Loss_l1/train_batch",
                    loss_mean_l1,
                    epoch * len(train_dataloader) + i,
                )
                writer.add_scalar(
                    "Loss_l2/train_batch",
                    loss_mean_l2,
                    epoch * len(train_dataloader) + i,
                )

            logging.info(
                f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] Batch [{i + 1}/{len(train_dataloader)}] "
                + f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}"
            )

        # save model, optimizer, and scaler
        if (opt.checkpoint_interval != -1) and (i % opt.checkpoint_interval_batch == 0):
            torch.save(
                model.state_dict(),
                opt.results_dir
                + "/saved_models/%s/model_%d_batch_%d.pth" % (opt.exp_name, epoch, i),
            )
            torch.save(
                optimizer.state_dict(),
                opt.results_dir
                + "/saved_models/%s/optimizer_%d_batch_%d.pth"
                % (opt.exp_name, epoch, i),
            )
            if opt.use_amp:
                torch.save(
                    scaler.state_dict(),
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d_batch_%d.pth"
                    % (opt.exp_name, epoch, i),
                )

    if track_gradients:
        return loss_list, loss_list_l1, loss_list_l2, grads
    else:
        return loss_list, loss_list_l1, loss_list_l2


def basic_train(opt, rng=None, epochs=3):

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
        use_splatting=opt.use_splatting,
    )

    model = SUPPORT(
        in_channels=opt.input_frames,
        mid_channels=opt.unet_channels,
        depth=opt.depth,
        blind_conv_channels=opt.blind_conv_channels,
        one_by_one_channels=opt.one_by_one_channels,
        last_layer_channels=opt.last_layer_channels,
        bs_size=opt.bs_size,
        bp=opt.bp,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
    rng = np.random.default_rng(rng)

    outs = []
    for epoch in range(epochs):
        outsed = train(
            dataloader_train,
            model,
            optimizer,
            scaler,
            rng,
            None,
            epoch,
            opt,
        )
        print(f"Epoch {epoch} loss: {np.mean(outsed[0][-10:])}")
        outs.append(outsed)

    return outs, model


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = parse_arguments()
    cuda = torch.cuda.is_available() and (not opt.use_CPU)
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    rng = np.random.default_rng(opt.random_seed)

    os.makedirs(opt.results_dir + "/images/{}".format(opt.exp_name), exist_ok=True)
    os.makedirs(
        opt.results_dir + "/saved_models/{}".format(opt.exp_name), exist_ok=True
    )
    os.makedirs(opt.results_dir + "/logs".format(opt.exp_name), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=opt.results_dir + "/logs/{}.log".format(opt.exp_name),
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    writer = SummaryWriter(opt.results_dir + "/tsboard/{}".format(opt.exp_name))

    # -----------
    # Dataset
    # ----------
    data_dir = "/gpfs/home/warnet02/data/stephen/run012"
    noisy_data = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    opt.noisy_data = noisy_data
    dataloader_train = gen_train_dataloader(
        opt.patch_size,
        opt.patch_interval,
        opt.batch_size,
        noisy_data,
        opt,
        is_zarr=opt.is_zarr,
    )

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    # Load splatting config if enabled
    splatting_config = None
    point_offset_config = None
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
        logging.info(f"Loaded splatting parameters from {opt.splatting_params_json}")

        if opt.use_point_offset:
            point_offset_config = {
                "hidden_channels": opt.point_offset_hidden_channels,
                "max_offset_px": opt.point_offset_max_px,
            }

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
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Initialize GradScaler if AMP is enabled
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    if cuda:
        model = model.cuda()

    if opt.epoch != 0:
        model.load_state_dict(
            torch.load(
                opt.results_dir
                + "/saved_models/%s/model_%d.pth" % (opt.exp_name, opt.epoch - 1)
            )
        )
        optimizer.load_state_dict(
            torch.load(
                opt.results_dir
                + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, opt.epoch - 1)
            )
        )
        if opt.use_amp:
            scaler.load_state_dict(
                torch.load(
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d.pth" % (opt.exp_name, opt.epoch - 1)
                )
            )
        print(
            "Loaded pre-trained model and optimizer weights of epoch {}".format(
                opt.epoch - 1
            )
        )

    # ----------
    # Training & Validation
    # ----------
    for epoch in range(opt.epoch, opt.n_epochs):
        dataloader_train.dataset.precompute_indices()

        # Train returns different things based on splatting mode
        if opt.use_splatting:
            loss_list, loss_list_components = train(
                dataloader_train, model, optimizer, scaler, rng, writer, epoch, opt
            )
        else:
            loss_list, loss_list_l1, loss_list_l2 = train(
                dataloader_train, model, optimizer, scaler, rng, writer, epoch, opt
            )

        # logging
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if epoch % opt.logging_interval == 0:
            loss_mean = np.mean(np.array(loss_list))

            if writer is not None:
                writer.add_scalar("Loss/train", loss_mean, epoch)

            if opt.use_splatting:
                # Log component losses
                component_means = {
                    key: np.mean(np.array(vals))
                    for key, vals in loss_list_components.items()
                }
                if writer is not None:
                    for key, val in component_means.items():
                        writer.add_scalar(f"Loss_Components/{key}", val, epoch)

                logging.info(
                    f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "
                    + f"loss: {loss_mean:.4f}, denoise: {component_means['denoise']:.4f}, "
                    + f"param_reg: {component_means['param_reg']:.4f}"
                )

                # Log splatting parameters
                if writer is not None:
                    validate_splatting_params(model, writer, epoch)
            else:
                loss_mean_l1 = np.mean(np.array(loss_list_l1))
                loss_mean_l2 = np.mean(np.array(loss_list_l2))
                if writer is not None:
                    writer.add_scalar("Loss_l1/train", loss_mean_l1, epoch)
                    writer.add_scalar("Loss_l2/train", loss_mean_l2, epoch)
                logging.info(
                    f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "
                    + f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}"
                )

        # Visualize splatting output periodically
        if opt.use_splatting and (epoch % opt.sample_interval == 0):
            # Get a sample batch for visualization
            sample_batch = next(iter(dataloader_train))[0].cuda()
            visualize_splatting_output(
                model,
                sample_batch,
                opt.results_dir + "/images/{}".format(opt.exp_name),
                epoch,
            )

        if (opt.checkpoint_interval != -1) and (epoch % opt.checkpoint_interval == 0):
            torch.save(
                model.state_dict(),
                opt.results_dir
                + "/saved_models/%s/model_%d.pth" % (opt.exp_name, epoch),
            )
            torch.save(
                optimizer.state_dict(),
                opt.results_dir
                + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, epoch),
            )
            if opt.use_amp:
                torch.save(
                    scaler.state_dict(),
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d.pth" % (opt.exp_name, epoch),
                )

        # if (epoch % opt.sample_interval == 0):
        #     skio.imsave(opt.results_dir + "/images/%s/denoised_%d.pth" % (opt.exp_name, epoch), )
