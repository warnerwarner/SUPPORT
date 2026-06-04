"""
Compare SUPPORT splatting reconstruction against EO reference implementation.

This test reconstructs a small temporal slice from the same zarr dataset using:
1) SUPPORT LearnableSplattingStage (current implementation)
2) EO reference splatting._splat_frame logic

and reports error metrics.
"""

import json
import importlib.util
import sys
import numpy as np
import torch
import zarr

project_root = "/gpfs/data/shohamlab/tom/support"
if project_root not in sys.path:
    sys.path.append(project_root)
from model.splatting_stage import LearnableSplattingStage


def _load_eo_modules():
    eo_dir = "/gpfs/home/warnet02/data/eo_accelerated_2p"
    if eo_dir not in sys.path:
        sys.path.insert(0, eo_dir)

    utils_spec = importlib.util.spec_from_file_location(
        "eo_utils", f"{eo_dir}/utils.py"
    )
    eo_utils = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(eo_utils)

    splat_spec = importlib.util.spec_from_file_location(
        "eo_splatting", f"{eo_dir}/splatting.py"
    )
    eo_splatting = importlib.util.module_from_spec(splat_spec)
    splat_spec.loader.exec_module(eo_splatting)

    return eo_splatting


def _eo_reconstruct(series, params, y_offset=0, x_offset=0, w_global=None):
    # series: [T, 2, H, W]
    spl = params["splatting"]
    n_frames, _, n_lines, n_line = series.shape

    n_y = int(n_lines * spl["eod_n_spots"] * spl["splat_grid_upsample_factor"])
    n_x = int(n_y * spl["scan_aspect"])

    if w_global is None:
        w_global = n_line

    n_modeled_line = int(w_global / spl["reso_temporal_fill_fraction"])
    n_blanked = n_modeled_line - w_global
    n_total = int(n_lines * n_modeled_line)

    spatial_fill_fraction = np.sin(spl["reso_temporal_fill_fraction"] * np.pi / 2)
    t = np.arange(n_total)
    y = np.linspace(0, n_y, n_total)
    x = np.cos(np.pi / n_modeled_line * t) / spatial_fill_fraction
    x = (x + 1.0) / 2.0 * n_x

    odd_mask = (
        np.arange(n_line) + x_offset + n_blanked / 2 - spl["reso_phase_shift_px"][0]
    ).astype(int)
    even_mask = (
        np.arange(n_line) + x_offset + n_blanked / 2 + spl["reso_phase_shift_px"][1]
    ).astype(int)
    odd_mask = np.clip(odd_mask, 0, n_modeled_line - 1)
    even_mask = np.clip(even_mask, 0, n_modeled_line - 1)

    scan_model = np.zeros((n_lines, n_line, 2), dtype=np.float32)
    for i in range(n_lines):
        global_i = y_offset + i
        if global_i % 2 == 0:
            scan_model[i, :, 0] = y[even_mask + i * n_modeled_line]
            scan_model[i, :, 1] = x[even_mask + i * n_modeled_line]
        else:
            scan_model[i, :, 0] = y[odd_mask + i * n_modeled_line]
            scan_model[i, :, 1] = x[odd_mask + i * n_modeled_line]

    eo_splat = _load_eo_modules()

    if spl.get("eod_intensity_correction", True):
        series = series.copy()
        series[:, 1, :, :] = eo_splat.bias_correction(
            series[:, 1, :, :].flatten(), series[:, 0, :, :].flatten(), eps=1e-2
        ).reshape(series[:, 1, :, :].shape)

    eod_amplitude_px = spl["eod_amplitude"] * n_y
    out = np.zeros(
        (
            n_frames,
            n_y // spl["splat_grid_upsample_factor"],
            n_x // spl["splat_grid_upsample_factor"],
        ),
        dtype=np.float32,
    )

    for i in range(n_frames):
        dst_hi = np.zeros((n_y, n_x), dtype=np.float32)
        eo_splat._splat_frame(
            frame=series[i],
            dst=dst_hi,
            scan_model=scan_model,
            eod_amplitude_px=eod_amplitude_px,
            eod_phase_rad=tuple(spl["eod_phase_rad"]),
            splat_sigma_px=spl["splat_sigma_px"],
            splat_grid_upsample_factor=int(spl["splat_grid_upsample_factor"]),
            impute_missing=False,
        )
        out[i] = dst_hi[: out.shape[1], : out.shape[2]]

    return out


def test_support_matches_eo_reconstruction():
    data_zarr = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00001.zarr"
    params_json = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/260430_splat_params.json"

    with open(params_json, "r") as f:
        params = json.load(f)

    with zarr.open(data_zarr, mode="r") as z:
        eod = z["eod"][:5].astype(np.float32)  # [T,H,W]
        pos = z["position"][:5].astype(np.float32)

    drop_lines = int(params["preprocessing"]["drop_lines"])
    series = np.stack([eod, pos], axis=1)  # [T,2,H,W]
    series = series[:, :, drop_lines:, :]

    # EO reference
    eo_out = _eo_reconstruct(series, params)  # [T,H',W']

    # SUPPORT
    stage = LearnableSplattingStage(initial_params=params, drop_lines=drop_lines).eval()
    x = torch.from_numpy(np.stack([eod, pos], axis=1)).unsqueeze(0)  # [1,T,2,H,W]
    with torch.no_grad():
        support_out = stage(x).squeeze(0).cpu().numpy()

    assert support_out.shape == eo_out.shape

    mae = np.mean(np.abs(support_out - eo_out))
    rmse = np.sqrt(np.mean((support_out - eo_out) ** 2))
    ref_scale = np.mean(np.abs(eo_out)) + 1e-8
    nmae = mae / ref_scale

    print(f"SUPPORT vs EO parity: MAE={mae:.6f}, RMSE={rmse:.6f}, NMAE={nmae:.6f}")

    # Parity threshold: support should numerically match EO implementation closely
    assert nmae < 0.10, f"Parity mismatch too large: NMAE={nmae:.6f}"


def test_crop_origin_matches_full_frame_reference():
    data_zarr = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00001.zarr"
    params_json = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/260430_splat_params.json"

    with open(params_json, "r") as f:
        params = json.load(f)

    with zarr.open(data_zarr, mode="r") as z:
        eod = z["eod"][:3].astype(np.float32)
        pos = z["position"][:3].astype(np.float32)

    drop_lines = int(params["preprocessing"]["drop_lines"])
    eod = eod[:, drop_lines:, :]
    pos = pos[:, drop_lines:, :]

    # Full frame EO reconstruction
    full_series = np.stack([eod, pos], axis=1)  # [T,2,H,W]
    eo_full = _eo_reconstruct(
        full_series, params, y_offset=0, x_offset=0, w_global=eod.shape[-1]
    )

    # Crop with non-zero origin
    init_h = 6
    init_w = 700
    crop_h = 16
    crop_w = 256
    eod_crop = eod[:, init_h : init_h + crop_h, init_w : init_w + crop_w]
    pos_crop = pos[:, init_h : init_h + crop_h, init_w : init_w + crop_w]

    stage = LearnableSplattingStage(initial_params=params, drop_lines=drop_lines).eval()
    x_crop = torch.from_numpy(np.stack([eod_crop, pos_crop], axis=1)).unsqueeze(0)
    origin = torch.tensor([[init_h, init_w]], dtype=torch.int64)

    with torch.no_grad():
        support_crop = (
            stage(
                x_crop,
                patch_origin=origin,
                full_n_samples=torch.tensor([eod.shape[-1]], dtype=torch.int64),
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # EO crop reconstruction with global crop origin offsets
    eo_crop = _eo_reconstruct(
        np.stack([eod_crop, pos_crop], axis=1),
        params,
        y_offset=init_h,
        x_offset=init_w,
        w_global=eod.shape[-1],
    )

    assert support_crop.shape == eo_crop.shape

    mae = np.mean(np.abs(support_crop - eo_crop))
    ref_scale = np.mean(np.abs(eo_crop)) + 1e-8
    nmae = mae / ref_scale
    print(f"Crop-origin parity: init_h={init_h}, init_w={init_w}, NMAE={nmae:.6f}")

    assert nmae < 0.12, f"Crop-origin parity mismatch too large: NMAE={nmae:.6f}"


if __name__ == "__main__":
    test_support_matches_eo_reconstruction()
