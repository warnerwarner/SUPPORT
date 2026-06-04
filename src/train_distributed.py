import os
import random
import logging
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import skimage.io as skio

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.dataset import gen_train_dataloader, random_transform
from src.utils.util import parse_arguments
from model.SUPPORT import SUPPORT


def setup_distributed():
    """
    Initialize distributed training environment from SLURM variables

    Returns:
        rank: Global rank across all nodes (0 to world_size-1)
        local_rank: Local rank on this node (0 to gpus_per_node-1)
        world_size: Total number of processes
    """
    # Get SLURM environment variables
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # Set the device for this process
    torch.cuda.set_device(local_rank)

    # Initialize process group
    # MASTER_ADDR and MASTER_PORT should be set in the SLURM script
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train(
    train_dataloader,
    model,
    optimizer,
    scaler,
    rng,
    writer,
    epoch,
    opt,
    rank,
    world_size,
):
    """
    Train a model for a single epoch (distributed version)

    Arguments:
        train_dataloader: (Pytorch DataLoader)
        model: (Pytorch nn.Module, wrapped with DDP)
        optimizer: (Pytorch optimzer)
        scaler: (GradScaler for AMP)
        rng: numpy random number generator
        writer: (Tensorboard writer, only on rank 0)
        epoch: epoch of training (int)
        opt: argparse dictionary
        rank: Global rank of this process
        world_size: Total number of processes

    Returns:
        loss_list: list of total loss of each batch ([float])
        loss_list_l1: list of L1 loss of each batch ([float])
        loss_list_l2: list of L2 loss of each batch ([float])
    """

    is_rotate = True if model.module.bs_size[0] == model.module.bs_size[1] else False

    # initialize
    model.train()
    loss_list_l1 = []
    loss_list_l2 = []
    loss_list = []

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    loss_coef = opt.loss_coef

    # Only show progress bar on rank 0
    dataloader_iter = tqdm(train_dataloader) if rank == 0 else train_dataloader

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
            phase_sin = None
            phase_cos = None

        B, T, X, Y = noisy_image.shape
        noisy_image = noisy_image.cuda()
        noisy_image, _, phase_sin, phase_cos = random_transform(
            noisy_image, None, rng, is_rotate, phase_sin=phase_sin, phase_cos=phase_cos
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

        # print log (only rank 0)
        if (
            rank == 0
            and (epoch % opt.logging_interval == 0)
            and (i % opt.logging_interval_batch == 0)
        ):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            writer.add_scalar(
                "Loss_l1/train_batch", loss_mean_l1, epoch * len(train_dataloader) + i
            )
            writer.add_scalar(
                "Loss_l2/train_batch", loss_mean_l2, epoch * len(train_dataloader) + i
            )
            writer.add_scalar(
                "Loss/train_batch", loss_mean, epoch * len(train_dataloader) + i
            )

            logging.info(
                f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] Batch [{i + 1}/{len(train_dataloader)}] "
                + f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f} "
                + f"[{world_size} GPUs]"
            )

        # save model, optimizer, and scaler (only rank 0)
        if (
            rank == 0
            and (opt.checkpoint_interval != -1)
            and (i % opt.checkpoint_interval_batch == 0)
        ):
            # Save the underlying model (unwrap DDP)
            torch.save(
                model.module.state_dict(),
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

    return loss_list, loss_list_l1, loss_list_l2


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # Initialize distributed training
    rank, local_rank, world_size = setup_distributed()

    # Only print from rank 0 to avoid spam
    if rank == 0:
        print(f"=" * 60)
        print(f"Initialized Distributed Training")
        print(f"=" * 60)
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")
        print(f"  Master addr: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"  Master port: {os.environ.get('MASTER_PORT', 'not set')}")
        print(f"  Node list: {os.environ.get('SLURM_NODELIST', 'not set')}")
        print(f"=" * 60)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = parse_arguments()
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    rng = np.random.default_rng(opt.random_seed + rank)  # Different seed per rank

    # Only rank 0 creates directories and sets up logging
    if rank == 0:
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
    else:
        writer = None
        # Suppress logging for non-rank-0 processes
        logging.basicConfig(level=logging.ERROR)

    # Barrier to ensure rank 0 has created directories before others proceed
    dist.barrier()

    # -----------
    # Dataset
    # ----------
    # IMPORTANT: Use gen_train_dataloader() to ensure proper data normalization
    # This function handles:
    #   1. Loading TIFF files with proper alignment (if is_raw=True)
    #   2. Rolling mean temporal smoothing (if rolling_mean != 1)
    #   3. Per-file normalization to mean=0, std=1 (CRITICAL for training!)
    # All ranks load the same data (DDP requirement)

    if rank == 0:
        print(f"Loading data with gen_train_dataloader()...")
        print(f"  Files: {len(opt.noisy_data)}")
        print(f"  is_raw: {opt.is_raw}")
        print(f"  is_zarr: {opt.is_zarr}")
        print(f"  rolling_mean: {opt.rolling_mean}")

    # Use the built-in dataloader generator which handles normalization correctly
    dataloader_train = gen_train_dataloader(
        opt.patch_size,
        opt.patch_interval,
        opt.batch_size,
        opt.noisy_data,  # Pass file paths, not pre-loaded tensors
        opt,
        is_zarr=opt.is_zarr,
        is_raw=opt.is_raw,
        rank=rank,  # Pass rank for distributed cache synchronization
        use_phase_conditioning=opt.use_phase_conditioning,
    )
    dist.barrier()
    if rank == 0:
        print("All ranks finished loading data, proceeding to model")

    # Now wrap the dataset with DistributedSampler for multi-GPU training
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader

    dataset_train = dataloader_train.dataset
    sampler = DistributedSampler(
        dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )

    # Recreate dataloader with distributed sampler (replaces shuffle=True)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=opt.n_cpu,
        pin_memory=True,
        prefetch_factor=opt.prefetch_factor,
    )
    if rank == 0:
        print("Waiting for all ranks to finish data loading")
    dist.barrier()

    if rank == 0:
        print("All ranks finished loading data!")
        print(f"\nDataset Information:")
        print(f"  Total dataset size: {len(dataset_train)}")
        print(f"  Samples per GPU: {len(dataset_train) // world_size}")
        print(f"  Batches per epoch (per GPU): {len(dataloader_train)}")
        print(f"  Batch size per GPU: {opt.batch_size}")
        print(f"  Effective batch size: {opt.batch_size * world_size}")

        # Verify normalization was applied
        if hasattr(dataset_train, "mean_images") and hasattr(
            dataset_train, "std_images"
        ):
            print(f"\n✓ Data normalization verified:")
            if (
                hasattr(dataset_train, "load_to_memory")
                and dataset_train.load_to_memory
            ):
                print(f"  Mean values: {dataset_train.mean_images.tolist()}")
                print(f"  Std values: {dataset_train.std_images.tolist()}")
                print(f"  ⚠ If these are NOT printed, data normalization FAILED!")
            else:
                print("Zarr loaded")
        else:
            print(
                f"\n✗ WARNING: Dataset does not have mean_images/std_images attributes!"
            )
            print(f"  This means data was NOT normalized - training will fail!")
        print()

    # ----------
    # Model, Optimizers, and Loss
    # ----------
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
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # Initialize GradScaler if AMP is enabled
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    if cuda:
        model = model.cuda(local_rank)
        # Wrap with DistributedDataParallel
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=opt.prevent_injection or opt.use_phase_conditioning,
        )
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel")
            print(f"Model device: cuda:{local_rank}")

    if opt.epoch != 0:
        # Load checkpoint (all ranks load the same checkpoint)
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        model.module.load_state_dict(
            torch.load(
                opt.results_dir
                + "/saved_models/%s/model_%d.pth" % (opt.exp_name, opt.epoch - 1),
                map_location=map_location,
            )
        )
        optimizer.load_state_dict(
            torch.load(
                opt.results_dir
                + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, opt.epoch - 1),
                map_location=map_location,
            )
        )
        if opt.use_amp:
            scaler.load_state_dict(
                torch.load(
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d.pth" % (opt.exp_name, opt.epoch - 1),
                    map_location=map_location,
                )
            )
        if rank == 0:
            print(
                "Loaded pre-trained model and optimizer weights of epoch {}".format(
                    opt.epoch - 1
                )
            )

    # ----------
    # Training & Validation
    # ----------
    if rank == 0:
        print(f"\nStarting training for {opt.n_epochs} epochs...")
        print(f"=" * 60)

    for epoch in range(opt.epoch, opt.n_epochs):
        # Set epoch for distributed sampler (important for proper shuffling)
        sampler.set_epoch(epoch)
        dataloader_train.dataset.precompute_indices()

        loss_list, loss_list_l1, loss_list_l2 = train(
            dataloader_train,
            model,
            optimizer,
            scaler,
            rng,
            writer,
            epoch,
            opt,
            rank,
            world_size,
        )

        # logging (only rank 0)
        if rank == 0:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            if epoch % opt.logging_interval == 0:
                loss_mean = np.mean(np.array(loss_list))
                loss_mean_l1 = np.mean(np.array(loss_list_l1))
                loss_mean_l2 = np.mean(np.array(loss_list_l2))

                writer.add_scalar("Loss/train", loss_mean, epoch)
                writer.add_scalar("Loss_l1/train", loss_mean_l1, epoch)
                writer.add_scalar("Loss_l2/train", loss_mean_l2, epoch)
                logging.info(
                    f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "
                    + f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}"
                )
                print(f"\n[{ts}] Epoch [{epoch}/{opt.n_epochs}] COMPLETE")
                print(
                    f"  Loss: {loss_mean:.4f}, L1: {loss_mean_l1:.4f}, L2: {loss_mean_l2:.4f}"
                )

            if (
                (opt.checkpoint_interval != -1)
                and (epoch % opt.checkpoint_interval == 0)
            ) or epoch == opt.n_epochs - 1:
                # Save underlying model (unwrap DDP)
                torch.save(
                    model.module.state_dict(),
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
                print(f"  Checkpoint saved: epoch {epoch}")

        # Barrier to ensure all ranks finish the epoch together
        dist.barrier()

    # Cleanup
    cleanup_distributed()
    if rank == 0:
        print(f"\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
