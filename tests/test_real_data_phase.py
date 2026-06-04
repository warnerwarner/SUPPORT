import torch
import numpy as np
import zarr
import sys
import os
from scipy.signal import hilbert

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.dataset import extract_phase
from model.SUPPORT import SUPPORT


def test_real_data_phase():
    zarr_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run011/stim_v1_fov3_440Hz_10X_2x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00033.zarr"
    print(f"Loading data from: {zarr_path}")

    try:
        z = zarr.open(zarr_path, mode="r")
        pos_data = z["position"]
        eod_data = z["eod"]
    except Exception as e:
        print(f"Error opening Zarr: {e}")
        return

    # Load a small chunk for testing (T=100, H=32, W=3606)
    t_slice = 100
    pos_chunk = pos_data[:t_slice, :, :]
    eod_chunk = eod_data[:t_slice, :, :]

    print(f"Position shape: {pos_chunk.shape}")
    print(f"EOD shape:      {eod_chunk.shape}")

    # Extract phase
    print("Extracting phase...")
    p_sin, p_cos = extract_phase(pos_chunk)

    # 1. Check Consistency (sin^2 + cos^2 = 1)
    mag_sq = p_sin**2 + p_cos**2
    mean_mag = np.mean(mag_sq)
    print(f"Verification - Mean (sin^2 + cos^2): {mean_mag:.6f}")
    assert np.allclose(mean_mag, 1.0, atol=1e-3), (
        f"Phase sin/cos magnitude is not 1.0 (got {mean_mag})"
    )

    # 2. Check accuracy against raw signal
    # The analytic signal phase at center should match the sinusoid at center
    center_idx = pos_chunk.shape[2] // 2
    raw_center = pos_chunk[:, :, center_idx]

    # Remove mean to check correlation properly
    raw_center_norm = raw_center - np.mean(raw_center)

    corr_cos = np.corrcoef(raw_center_norm.flatten(), p_cos.flatten())[0, 1]
    corr_sin = np.corrcoef(raw_center_norm.flatten(), p_sin.flatten())[0, 1]

    print(f"Correlation (Raw vs Cos): {corr_cos:.4f}")
    print(f"Correlation (Raw vs Sin): {corr_sin:.4f}")

    # One of these should be very high (typically Cos for the real part of Hilbert)
    max_corr = max(abs(corr_cos), abs(corr_sin))
    assert max_corr > 0.8, (
        f"Low correlation with raw signal: {max_corr:.4f}. Extraction may be inaccurate."
    )
    print(f"✓ Phase correlates strongly with raw signal (Max |R| = {max_corr:.4f})")

    # 3. Model Forward Pass with real shapes
    print("-" * 30)
    print("Testing model with real data shapes...")
    n_frames = 61
    patch_h = 16
    patch_w = 320

    model = SUPPORT(
        in_channels=n_frames,
        use_phase_conditioning=True,
        mid_channels=[16, 32],
        blind_conv_channels=16,
        one_by_one_channels=[16, 8],
        last_layer_channels=[8, 4],
        is_raw=True,
    )

    # Prepare dummy tensors matching real data slices
    x = torch.randn(1, n_frames, patch_h, patch_w)
    # Match the height of the patch (H=16)
    ps = torch.tensor(p_sin[:n_frames, :patch_h]).unsqueeze(0).float()
    pc = torch.tensor(p_cos[:n_frames, :patch_h]).unsqueeze(0).float()

    output = model(x, ps, pc)
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1, patch_h, patch_w)
    print("✓ Model forward pass with real data dimensions successful.")


if __name__ == "__main__":
    try:
        test_real_data_phase()
    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)
