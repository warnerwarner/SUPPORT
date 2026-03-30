"""
Tests to verify that phase information is correctly passed through the model.

These tests check:
1. Phase model instantiates correctly with expected extra parameters
2. Phase affects model output (functional test)
3. Non-phase model ignores phase (backward compatibility)
4. Per-row constancy of phase channels after broadcast
5. Gradients flow through phase inputs
"""

import torch
import sys
import os

# Add project root to path
project_root = "/gpfs/data/shohamlab/tom/support"
if project_root not in sys.path:
    sys.path.append(project_root)

from model.SUPPORT import SUPPORT


def test_phase_model_structure():
    """
    Test 1: Phase Model Structure

    Verify that the phase-conditioned model has the expected inject_proj layers
    and that the non-phase model does not.
    """
    print("\n" + "=" * 70)
    print("Test 1: Phase Model Structure")
    print("=" * 70)

    model_phase = SUPPORT(
        in_channels=61,
        mid_channels=[96, 192, 384, 768, 1536],
        depth=8,
        blind_conv_channels=96,
        one_by_one_channels=[48, 24],
        last_layer_channels=[96, 48, 24],
        bs_size=[2, 2],
        bp=False,
        is_raw=True,
        prevent_injection=False,
        use_phase_conditioning=True,
    )

    model_nophase = SUPPORT(
        in_channels=61,
        mid_channels=[96, 192, 384, 768, 1536],
        depth=8,
        blind_conv_channels=96,
        one_by_one_channels=[48, 24],
        last_layer_channels=[96, 48, 24],
        bs_size=[2, 2],
        bp=False,
        is_raw=True,
        prevent_injection=False,
        use_phase_conditioning=False,
    )

    # Phase model should have inject_proj ModuleLists
    assert hasattr(model_phase, "inject_proj_3x3"), "Missing inject_proj_3x3"
    assert hasattr(model_phase, "inject_proj_5x5"), "Missing inject_proj_5x5"
    print(f"  inject_proj_3x3 layers: {len(model_phase.inject_proj_3x3)}")
    print(f"  inject_proj_5x5 layers: {len(model_phase.inject_proj_5x5)}")

    # Each inject_proj should be a 1x1 conv from 3 channels to blind_conv_channels
    for i, proj in enumerate(model_phase.inject_proj_3x3):
        assert proj.in_channels == 3, (
            f"inject_proj_3x3[{i}] in_channels={proj.in_channels}, expected 3"
        )
        assert proj.out_channels == 96, (
            f"inject_proj_3x3[{i}] out_channels={proj.out_channels}, expected 96"
        )
        assert proj.kernel_size == (1, 1), (
            f"inject_proj_3x3[{i}] kernel_size={proj.kernel_size}, expected (1,1)"
        )

    # First blind conv in phase model should accept 3 input channels
    first_blind_3x3 = model_phase.blind_convs3x3[0]
    assert first_blind_3x3.in_channels == 3, (
        f"First blind conv in_channels={first_blind_3x3.in_channels}, expected 3"
    )
    print(f"  First blind conv (3x3) in_channels: {first_blind_3x3.in_channels}")

    # First blind conv in non-phase model should accept 1 input channel
    first_blind_nophase = model_nophase.blind_convs3x3[0]
    assert first_blind_nophase.in_channels == 1, (
        f"Non-phase first blind conv in_channels={first_blind_nophase.in_channels}, expected 1"
    )
    print(
        f"  First blind conv (3x3, no-phase) in_channels: {first_blind_nophase.in_channels}"
    )

    # Non-phase model should NOT have inject_proj
    assert not hasattr(model_nophase, "inject_proj_3x3"), (
        "Non-phase model shouldn't have inject_proj_3x3"
    )
    assert not hasattr(model_nophase, "inject_proj_5x5"), (
        "Non-phase model shouldn't have inject_proj_5x5"
    )
    print("  Non-phase model: no inject_proj layers (correct)")

    # Count extra parameters from phase conditioning
    phase_params = sum(p.numel() for p in model_phase.parameters())
    nophase_params = sum(p.numel() for p in model_nophase.parameters())
    extra = phase_params - nophase_params
    print(f"  Phase model params:    {phase_params:,}")
    print(f"  Non-phase model params: {nophase_params:,}")
    print(f"  Extra params from phase: {extra:,}")

    print("  ✓ PASS: Phase model structure is correct")


def test_phase_affects_output():
    """
    Test 2: Phase Affects Model Output

    Same image data with different phase tensors should produce different outputs.
    """
    print("\n" + "=" * 70)
    print("Test 2: Phase Affects Model Output")
    print("=" * 70)

    model = SUPPORT(
        in_channels=61,
        mid_channels=[96, 192, 384, 768, 1536],
        depth=8,
        blind_conv_channels=96,
        one_by_one_channels=[48, 24],
        last_layer_channels=[96, 48, 24],
        bs_size=[2, 2],
        bp=False,
        is_raw=True,
        prevent_injection=False,
        use_phase_conditioning=True,
    ).cuda()

    # Same image data, two different phase tensors
    x = torch.randn(2, 61, 16, 320).cuda()
    phase_sin_a = torch.randn(2, 61, 16).cuda()
    phase_cos_a = torch.randn(2, 61, 16).cuda()
    phase_sin_b = torch.randn(2, 61, 16).cuda()
    phase_cos_b = torch.randn(2, 61, 16).cuda()

    model.eval()
    with torch.no_grad():
        out_a = model(x, phase_sin_a, phase_cos_a)
        out_b = model(x, phase_sin_b, phase_cos_b)

    # Outputs should differ
    max_diff = (out_a - out_b).abs().max().item()
    mean_diff = (out_a - out_b).abs().mean().item()

    print(f"  Image shape: {x.shape}")
    print(f"  Phase shape: {phase_sin_a.shape}")
    print(f"  Output shape: {out_a.shape}")
    print(f"  Max diff between outputs: {max_diff:.6f}")
    print(f"  Mean diff between outputs: {mean_diff:.6f}")

    assert not torch.allclose(out_a, out_b, atol=1e-6), "Phase has no effect on output!"
    assert max_diff > 1e-4, f"Phase effect too small: max_diff={max_diff}"

    print("  ✓ PASS: Phase affects model output")


def test_no_phase_model_ignores_phase():
    """
    Test 3: No Phase = Original Behavior

    With use_phase_conditioning=False, model should ignore phase arguments.
    """
    print("\n" + "=" * 70)
    print("Test 3: Non-Phase Model Ignores Phase")
    print("=" * 70)

    model_nophase = SUPPORT(
        in_channels=61,
        mid_channels=[96, 192, 384, 768, 1536],
        depth=8,
        blind_conv_channels=96,
        one_by_one_channels=[48, 24],
        last_layer_channels=[96, 48, 24],
        bs_size=[2, 2],
        bp=False,
        is_raw=True,
        prevent_injection=False,
        use_phase_conditioning=False,
    ).cuda()

    x = torch.randn(2, 61, 16, 320).cuda()
    phase_sin_a = torch.randn(2, 61, 16).cuda()
    phase_cos_a = torch.randn(2, 61, 16).cuda()

    model_nophase.eval()
    with torch.no_grad():
        out_none = model_nophase(x, None, None)
        out_with = model_nophase(x, phase_sin_a, phase_cos_a)

    max_diff = (out_none - out_with).abs().max().item()

    print(f"  Image shape: {x.shape}")
    print(f"  Output with phase=None: {out_none.shape}")
    print(f"  Output with phase tensors: {out_with.shape}")
    print(f"  Max diff: {max_diff:.10f}")

    assert torch.allclose(out_none, out_with, atol=1e-6), (
        "Non-phase model should ignore phase!"
    )

    print("  ✓ PASS: Non-phase model correctly ignores phase arguments")


def test_per_row_constancy():
    """
    Test 4: Per-Row Constancy of Phase Channels

    After broadcast, phase channels should be constant across W dimension.
    """
    print("\n" + "=" * 70)
    print("Test 4: Per-Row Constancy of Phase Channels")
    print("=" * 70)

    model = SUPPORT(
        in_channels=61,
        mid_channels=[96, 192, 384, 768, 1536],
        depth=8,
        blind_conv_channels=96,
        one_by_one_channels=[48, 24],
        last_layer_channels=[96, 48, 24],
        bs_size=[2, 2],
        bp=False,
        is_raw=True,
        prevent_injection=False,
        use_phase_conditioning=True,
    ).cuda()

    x = torch.randn(2, 61, 16, 320).cuda()
    phase_sin = torch.randn(2, 61, 16).cuda()
    phase_cos = torch.randn(2, 61, 16).cuda()

    # Hook to capture x_inject inside forward_bsnet
    # The hook is on blind_convs3x3[0] (a Conv2d). At layer 0, x1 = x_inject,
    # so the conv's input[0] IS x_inject: (B, 3, H, W) with channels [frame, sin, cos].
    x_inject_captured = None

    def hook_fn(module, input, output):
        nonlocal x_inject_captured
        x_inject_captured = input[0]  # Conv2d input is just (tensor,)

    # Register hook on the first blind conv to capture its input (which is x_inject)
    handle = model.blind_convs3x3[0].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(x, phase_sin, phase_cos)

    handle.remove()

    # Check per-row constancy
    assert x_inject_captured is not None, "Failed to capture x_inject"
    assert x_inject_captured.shape[1] == 3, (
        f"x_inject should have 3 channels, got {x_inject_captured.shape[1]}"
    )

    # Check sin channel (channel 1) is constant across W for each row
    sin_channel = x_inject_captured[:, 1, :, :]  # (B, H, W)
    for b in range(sin_channel.shape[0]):
        for h in range(sin_channel.shape[1]):
            row_values = sin_channel[b, h, :]
            assert torch.allclose(
                row_values, row_values[0].expand_as(row_values), atol=1e-6
            ), f"Sin channel not constant across W at batch={b}, row={h}"

    # Check cos channel (channel 2) is constant across W for each row
    cos_channel = x_inject_captured[:, 2, :, :]  # (B, H, W)
    for b in range(cos_channel.shape[0]):
        for h in range(cos_channel.shape[1]):
            row_values = cos_channel[b, h, :]
            assert torch.allclose(
                row_values, row_values[0].expand_as(row_values), atol=1e-6
            ), f"Cos channel not constant across W at batch={b}, row={h}"

    print(f"  x_inject shape: {x_inject_captured.shape}")
    print(
        f"  Checked sin channel: constant across W for all {sin_channel.shape[0] * sin_channel.shape[1]} rows"
    )
    print(
        f"  Checked cos channel: constant across W for all {cos_channel.shape[0] * cos_channel.shape[1]} rows"
    )
    print("  ✓ PASS: Phase channels are per-row constant across W dimension")


def test_gradients_flow_through_phase():
    """
    Test 5: Gradients Flow Through Phase

    Phase inputs should receive gradients during backprop.
    """
    print("\n" + "=" * 70)
    print("Test 5: Gradients Flow Through Phase")
    print("=" * 70)

    model = SUPPORT(
        in_channels=61,
        mid_channels=[96, 192, 384, 768, 1536],
        depth=8,
        blind_conv_channels=96,
        one_by_one_channels=[48, 24],
        last_layer_channels=[96, 48, 24],
        bs_size=[2, 2],
        bp=False,
        is_raw=True,
        prevent_injection=False,
        use_phase_conditioning=True,
    ).cuda()

    # Create directly on CUDA so they remain leaf tensors (`.cuda()` on a
    # requires_grad tensor produces a non-leaf, whose .grad is never populated).
    phase_sin = torch.randn(2, 61, 16, device="cuda", requires_grad=True)
    phase_cos = torch.randn(2, 61, 16, device="cuda", requires_grad=True)
    x = torch.randn(2, 61, 16, 320, device="cuda")

    model.train()
    out = model(x, phase_sin, phase_cos)
    loss = out.sum()
    loss.backward()

    assert phase_sin.grad is not None, "No gradient on phase_sin!"
    assert phase_cos.grad is not None, "No gradient on phase_cos!"

    sin_grad_norm = phase_sin.grad.norm().item()
    cos_grad_norm = phase_cos.grad.norm().item()
    sin_grad_nonzero = (phase_sin.grad.abs() > 1e-8).sum().item()
    cos_grad_nonzero = (phase_cos.grad.abs() > 1e-8).sum().item()

    print(f"  Phase_sin grad norm: {sin_grad_norm:.6f}")
    print(f"  Phase_cos grad norm: {cos_grad_norm:.6f}")
    print(f"  Phase_sin nonzero grad elements: {sin_grad_nonzero}/{phase_sin.numel()}")
    print(f"  Phase_cos nonzero grad elements: {cos_grad_nonzero}/{phase_cos.numel()}")

    assert sin_grad_norm > 0, "Phase_sin gradients are all zero!"
    assert cos_grad_norm > 0, "Phase_cos gradients are all zero!"
    assert sin_grad_nonzero > 0, "Phase_sin has no nonzero gradients!"
    assert cos_grad_nonzero > 0, "Phase_cos has no nonzero gradients!"

    print("  ✓ PASS: Gradients flow through phase inputs")


def test_real_data_end_to_end():
    """
    Test 6: Real Data End-to-End

    Load actual zarr data, extract phase, build dataloader, pull a batch,
    verify shapes and phase properties, and run a forward+backward pass.
    """
    import numpy as np
    from src.utils.dataset import gen_train_dataloader, random_transform
    from src.utils.util import parse_arguments

    print("\n" + "=" * 70)
    print("Test 6: Real Data End-to-End")
    print("=" * 70)

    # --- Step 1: Parse arguments to load real data ---
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
        ]
    )

    print(f"  Data path: {opt.noisy_data}")
    print(f"  Patch size: {opt.patch_size}")
    print(f"  Phase conditioning: {opt.use_phase_conditioning}")

    # --- Step 2: Build dataloader with phase extraction ---
    print("  Loading data and extracting phase...")
    dataloader_train = gen_train_dataloader(
        opt.patch_size,
        opt.patch_interval,
        opt.batch_size,
        opt.noisy_data,
        opt,
        is_zarr=opt.is_zarr,
        is_raw=opt.is_raw,
        rank=0,
        use_phase_conditioning=opt.use_phase_conditioning,
    )

    print(f"  Dataset size: {len(dataloader_train.dataset)} patches")

    # --- Step 3: Pull one batch and check shapes ---
    print("  Pulling first batch...")
    batch = next(iter(dataloader_train))

    # With phase conditioning + zarr: 7-element tuple
    assert len(batch) == 7, f"Expected 7-element batch tuple, got {len(batch)}"
    noisy_image, _, ds_idx, noisy_image_avg, noisy_image_std, phase_sin, phase_cos = (
        batch
    )

    B, T, H, W = noisy_image.shape
    print(f"  noisy_image shape: ({B}, {T}, {H}, {W})")
    print(f"  phase_sin shape:   {phase_sin.shape}")
    print(f"  phase_cos shape:   {phase_cos.shape}")

    assert T == opt.patch_size[0], f"T={T}, expected {opt.patch_size[0]}"
    assert H == opt.patch_size[1], f"H={H}, expected {opt.patch_size[1]}"
    assert W == opt.patch_size[2], f"W={W}, expected {opt.patch_size[2]}"
    assert phase_sin.shape == (B, T, H), (
        f"phase_sin shape {phase_sin.shape}, expected ({B}, {T}, {H})"
    )
    assert phase_cos.shape == (B, T, H), (
        f"phase_cos shape {phase_cos.shape}, expected ({B}, {T}, {H})"
    )
    print("  ✓ Batch shapes correct")

    # --- Step 4: Phase unit circle property (sin^2 + cos^2 ≈ 1) ---
    mag_sq = phase_sin**2 + phase_cos**2
    mean_mag = mag_sq.mean().item()
    max_dev = (mag_sq - 1.0).abs().max().item()
    print(f"  Mean(sin^2 + cos^2): {mean_mag:.6f}")
    print(f"  Max |sin^2 + cos^2 - 1|: {max_dev:.6f}")
    assert abs(mean_mag - 1.0) < 0.01, (
        f"Phase magnitude not unit circle: mean={mean_mag}"
    )
    print("  ✓ Phase lies on unit circle")

    # --- Step 5: Apply random_transform and verify phase is flipped consistently ---
    rng = np.random.default_rng(42)
    is_rotate = opt.bs_size[0] == opt.bs_size[1]
    noisy_image_cuda = noisy_image.cuda()
    phase_sin_cuda = phase_sin.cuda()
    phase_cos_cuda = phase_cos.cuda()

    img_t, _, ps_t, pc_t = random_transform(
        noisy_image_cuda, None, rng, is_rotate, phase_sin_cuda, phase_cos_cuda
    )
    assert img_t.shape == noisy_image_cuda.shape, "Image shape changed after transform"
    assert ps_t.shape == phase_sin_cuda.shape, "Phase_sin shape changed after transform"
    assert pc_t.shape == phase_cos_cuda.shape, "Phase_cos shape changed after transform"
    # Phase should still be on unit circle after transform
    mag_sq_t = ps_t**2 + pc_t**2
    assert (mag_sq_t - 1.0).abs().max().item() < 0.01, (
        "Phase not on unit circle after transform"
    )
    print("  ✓ random_transform preserves shapes and unit circle property")

    # --- Step 6: Normalize and run forward pass ---
    noisy_image_avg = noisy_image_avg.reshape(-1, 1, 1, 1).cuda()
    noisy_image_std = noisy_image_std.reshape(-1, 1, 1, 1).cuda()
    img_norm = (img_t - noisy_image_avg) / noisy_image_std

    print("  Building model...")
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

    # Forward pass
    print("  Running forward pass...")
    model.train()
    output = model(img_norm, ps_t, pc_t)
    assert output.shape == (B, 1, H, W), (
        f"Output shape {output.shape}, expected ({B}, 1, {H}, {W})"
    )
    print(f"  Output shape: {output.shape}")
    print("  ✓ Forward pass successful")

    # --- Step 7: Backward pass ---
    print("  Running backward pass...")
    target = img_norm[:, T // 2, :, :].unsqueeze(1)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    print(f"  Loss: {loss.item():.6f}")

    # Verify gradients exist on model parameters
    n_params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_params_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Params with nonzero grad: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad > 0, "No parameters received gradients!"

    # Verify inject_proj layers specifically received gradients
    for i, proj in enumerate(model.inject_proj_3x3):
        assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0, (
            f"inject_proj_3x3[{i}] received no gradient"
        )
    for i, proj in enumerate(model.inject_proj_5x5):
        assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0, (
            f"inject_proj_5x5[{i}] received no gradient"
        )
    print("  ✓ Backward pass successful, inject_proj layers receive gradients")

    print("  ✓ PASS: Real data end-to-end test complete")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# Phase Structure Tests")
    print("#" * 70)

    try:
        test_phase_model_structure()
        test_phase_affects_output()
        test_no_phase_model_ignores_phase()
        test_per_row_constancy()
        test_gradients_flow_through_phase()
        test_real_data_end_to_end()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

    except AssertionError as e:  # noqa: F821 — intentional catch-all via Exception below
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
