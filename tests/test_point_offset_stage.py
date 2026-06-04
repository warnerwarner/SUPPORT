import torch

from model.SUPPORT import SUPPORT


def test_support_without_point_offset_still_works():
    model = SUPPORT(
        in_channels=21,
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=[1, 3],
        bp=False,
        use_splatting=False,
        use_point_offset=False,
    )

    x = torch.randn(1, 21, 32, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 32, 64)


def test_support_with_point_offset_and_splatting_runs():
    # Minimal valid splatting config for module construction
    splatting_config = {
        "initial_params": {
            "preprocessing": {"drop_lines": 1},
            "splatting": {
                "eod_n_spots": 15,
                "eod_amplitude": -0.025,
                "eod_phase_rad": [-0.5, 0.5],
                "reso_phase_shift_px": [0.0, 0.0],
                "reso_temporal_fill_fraction": 0.9,
                "scan_aspect": 3.0,
                "splat_sigma_px": 0.7,
                "splat_grid_upsample_factor": 2,
                "eod_intensity_correction": False,
            },
        },
        "drop_lines": 1,
    }

    model = SUPPORT(
        in_channels=21,
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=[1, 3],
        bp=False,
        use_splatting=True,
        splatting_config=splatting_config,
        use_point_offset=True,
        point_offset_config={"hidden_channels": 8, "max_offset_px": 1.0},
    )

    x = torch.randn(1, 21, 2, 16, 128)
    patch_origin = torch.tensor([[0, 0]], dtype=torch.int64)
    full_n_samples = torch.tensor([128], dtype=torch.int64)

    with torch.no_grad():
        y = model(x, patch_origin=patch_origin, full_n_samples=full_n_samples)

    assert y.ndim == 4
    assert y.shape[0] == 1
    assert y.shape[1] == 1
