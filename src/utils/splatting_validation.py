"""
Validation utilities for monitoring splatting quality during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import zarr


def validate_splatting_params(model, writer, epoch):
    """
    Log splatting parameters and their deviation from initial values.

    Args:
        model: SUPPORT model with splatting stage
        writer: TensorBoard writer
        epoch: Current epoch number
    """
    if not hasattr(model, "splatting_stage") or model.splatting_stage is None:
        return

    # Get current parameter values
    param_dict = model.splatting_stage.get_params_dict()
    initial_params = model.splatting_stage.initial_params_tensor.cpu().numpy()

    param_names = [
        "eod_amplitude",
        "eod_phase_rad_even",
        "eod_phase_rad_odd",
        "reso_phase_shift_even",
        "reso_phase_shift_odd",
        "splat_sigma",
        "reso_fill_fraction",
        "scan_aspect",
    ]

    # Log each parameter
    for i, (name, value) in enumerate(param_dict.items()):
        writer.add_scalar(f"SplatParams/{name}", value, epoch)

        # Compute deviation from initial value
        initial_value = initial_params[i]
        deviation = abs(value - initial_value) / (abs(initial_value) + 1e-8)
        writer.add_scalar(f"SplatParams/{name}_deviation", deviation, epoch)

    print(f"Epoch {epoch} - Splatting Parameters:")
    for name, value in param_dict.items():
        print(f"  {name}: {value:.6f}")


def visualize_splatting_output(model, sample_batch, output_path, epoch):
    """
    Create visualization of raw data, splatted output, and denoised result.

    Args:
        model: SUPPORT model
        sample_batch: [B, T, 2, H, W] raw batch (just use first item)
        output_path: Path to save visualization
        epoch: Current epoch number
    """
    model.eval()

    with torch.no_grad():
        # Take first sample and center frame
        raw_sample = sample_batch[0:1]  # [1, T, 2, H, W]
        T = raw_sample.shape[1]
        center_idx = T // 2

        # Get raw data
        raw_eod = raw_sample[0, center_idx, 0].cpu().numpy()  # [H, W]
        raw_pos = raw_sample[0, center_idx, 1].cpu().numpy()  # [H, W]

        # Get splatted output
        if hasattr(model, "splatting_stage") and model.splatting_stage is not None:
            splatted = model.splatting_stage(raw_sample)  # [1, T, H', W']
            splatted_frame = splatted[0, center_idx].cpu().numpy()  # [H', W']
        else:
            splatted_frame = None

        # Get denoised output
        denoised = model(raw_sample)  # [1, 1, H', W']
        denoised_frame = denoised[0, 0].cpu().numpy()  # [H', W']

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Raw EOD
    im0 = axes[0, 0].imshow(raw_eod, cmap="gray")
    axes[0, 0].set_title(f"Raw EOD (Epoch {epoch})")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Raw Position
    im1 = axes[0, 1].imshow(raw_pos, cmap="viridis")
    axes[0, 1].set_title(f"Raw Position Signal (Epoch {epoch})")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Splatted
    if splatted_frame is not None:
        im2 = axes[1, 0].imshow(splatted_frame, cmap="gray")
        axes[1, 0].set_title(f"Splatted Output (Epoch {epoch})")
        axes[1, 0].axis("off")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].text(0.5, 0.5, "No Splatting", ha="center", va="center")
        axes[1, 0].axis("off")

    # Denoised
    im3 = axes[1, 1].imshow(denoised_frame, cmap="gray")
    axes[1, 1].set_title(f"Denoised Output (Epoch {epoch})")
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/visualization_epoch{epoch:04d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    model.train()


def log_splatting_losses(loss_dict, writer, step):
    """
    Log individual loss components to TensorBoard.

    Args:
        loss_dict: Dict with loss components
        writer: TensorBoard writer
        step: Global step number
    """
    for key, value in loss_dict.items():
        if key != "total":
            writer.add_scalar(f"Loss_Components/{key}", value, step)
    writer.add_scalar("Loss/total", loss_dict["total"], step)
