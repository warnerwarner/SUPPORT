"""
Loss functions and utilities for training with learnable splatting stage.
"""

import torch
import torch.nn.functional as F


def compute_splatting_loss(model, raw_batch, denoised, epoch_factor, opt):
    """
    Compute multi-component loss for training with splatting stage.

    Args:
        model: SUPPORT model with splatting stage
        raw_batch: [B, T, 2, H, W] raw input (eod, position)
        denoised: [B, 1, H', W'] denoised output from model
        epoch_factor: float, decay factor for parameter regularization
        opt: options object with loss weights

    Returns:
        dict with 'total' loss and individual components
    """
    B, T, C, H, W = raw_batch.shape

    # Get splatted intermediate from the cached forward pass
    splatted = model._last_splatted  # [B, T, H', W']

    # 1. Main loss: Blind-spot denoising (L1)
    center_idx = T // 2
    target = splatted[:, center_idx : center_idx + 1, :, :].detach()
    loss_denoise = F.l1_loss(denoised, target)

    # 2. Parameter regularization (keep params near initialization)
    if hasattr(model, "splatting_stage") and model.splatting_stage is not None:
        current_params = model.splatting_stage.get_params_tensor()
        initial_params = model.splatting_stage.initial_params_tensor
        # Relative squared error
        loss_param_reg = torch.mean(
            ((current_params - initial_params) / (torch.abs(initial_params) + 1e-8))
            ** 2
        )
    else:
        loss_param_reg = torch.tensor(0.0, device=denoised.device)

    # 3. Signal preservation (mean/variance should match raw signal)
    raw_signal = raw_batch[:, :, 0, :, :]  # eod channel [B, T, H, W]
    loss_signal_mean = (splatted.mean() - raw_signal.mean()) ** 2
    loss_signal_var = (splatted.std() - raw_signal.std()) ** 2
    loss_signal_pres = loss_signal_mean + loss_signal_var

    # 4. Total variation (smoothness regularization)
    # Penalize sharp discontinuities in splatted output
    loss_tv = torch.mean(
        torch.abs(splatted[:, :, :, 1:] - splatted[:, :, :, :-1])
    ) + torch.mean(torch.abs(splatted[:, :, 1:, :] - splatted[:, :, :-1, :]))

    # 5. Temporal consistency (adjacent frames should be similar)
    if T > 1:
        loss_temporal = F.mse_loss(splatted[:, 1:], splatted[:, :-1])
    else:
        loss_temporal = torch.tensor(0.0, device=denoised.device)

    # 6. Offset regularization (if point-offset stage enabled)
    if hasattr(model, "_last_offsets") and model._last_offsets is not None:
        loss_offset_reg = torch.mean(model._last_offsets**2)
    else:
        loss_offset_reg = torch.tensor(0.0, device=denoised.device)

    # Combine with weights (parameter reg decays over epochs)
    total_loss = (
        opt.loss_weight_denoise * loss_denoise
        + opt.loss_weight_param_reg * epoch_factor * loss_param_reg
        + opt.loss_weight_signal_pres * loss_signal_pres
        + opt.loss_weight_tv * loss_tv
        + opt.loss_weight_temporal * loss_temporal
        + getattr(opt, "loss_weight_offset_reg", 0.1) * loss_offset_reg
    )

    return {
        "total": total_loss,
        "denoise": loss_denoise,
        "param_reg": loss_param_reg,
        "signal_pres": loss_signal_pres,
        "tv": loss_tv,
        "temporal": loss_temporal,
        "offset_reg": loss_offset_reg,
    }


def total_variation_loss(x):
    """
    Compute total variation loss for image smoothness.

    Args:
        x: [B, T, H, W] or [B, H, W] tensor

    Returns:
        scalar TV loss
    """
    if x.ndim == 4:
        # [B, T, H, W]
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    elif x.ndim == 3:
        # [B, H, W]
        tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

    return tv_h + tv_w
