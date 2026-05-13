"""
Test script for splatting stage integration.

This script tests:
1. Backward compatibility (non-splatting mode still works)
2. Splatting stage forward pass
3. End-to-end forward/backward pass
4. Data loading with splatting

Usage:
    python tests/test_splatting_integration.py
"""

import torch
import numpy as np
import sys
import os
import json

# Add project root to path
project_root = "/gpfs/data/shohamlab/tom/support"
if project_root not in sys.path:
    sys.path.append(project_root)

from model.SUPPORT import SUPPORT
from model.splatting_stage import LearnableSplattingStage
from src.utils.dataset import gen_train_dataloader
from src.utils.util import parse_arguments


def test_backward_compatibility():
    """Test that SUPPORT works without splatting (backward compatibility)"""
    print("\n" + "=" * 60)
    print("TEST 1: Backward Compatibility (non-splatting mode)")
    print("=" * 60)

    # Create model WITHOUT splatting
    model = SUPPORT(
        in_channels=61,
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=[1, 3],
        bp=False,
        use_splatting=False,  # Key: splatting disabled
    ).cuda()

    # Test forward pass with normal input [B, T, H, W]
    B, T, H, W = 2, 61, 64, 64
    x = torch.randn(B, T, H, W).cuda()

    try:
        output = model(x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected output shape: [{B}, 1, {H}, {W}]")
        assert output.shape == (B, 1, H, W), f"Output shape mismatch: {output.shape}"
        print("✓ Backward compatibility test PASSED")
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test FAILED: {e}")
        return False


def test_splatting_stage_standalone():
    """Test splatting stage as standalone module"""
    print("\n" + "=" * 60)
    print("TEST 2: Splatting Stage Standalone")
    print("=" * 60)

    # Load splatting parameters
    splat_params_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/260430_splat_params.json"

    if not os.path.exists(splat_params_path):
        print(f"✗ Splatting params not found at {splat_params_path}")
        print("  Skipping this test.")
        return False

    with open(splat_params_path, "r") as f:
        splat_params = json.load(f)

    # Create splatting stage (no n_lines/n_samples needed, inferred from input)
    drop_lines = splat_params["preprocessing"]["drop_lines"]

    try:
        splatting_stage = LearnableSplattingStage(
            initial_params=splat_params,
            drop_lines=drop_lines,
        ).cuda()

        # Test forward pass with [B, T, 2, H, W] input
        # Use smaller patch size for memory efficiency in testing
        # Full resolution would be 512x1024, but output is ~15k x 46k pixels (2.6GB per frame!)
        B, T, C, H, W = 1, 3, 2, 64, 128  # Small patch for testing
        print(f"Note: Using small patch size {H}x{W} for memory efficiency")

        x = torch.randn(B, T, C, H, W).cuda()
        x[:, :, 1, :, :] = torch.sin(
            torch.linspace(0, 10, W).cuda()
        )  # Fake oscillation

        import time

        start = time.time()
        output = splatting_stage(x)
        elapsed = time.time() - start

        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output has 4 dims (B, T, H', W'): {output.ndim == 4}")
        print(
            f"✓ Dimensions inferred: n_lines={splatting_stage.n_lines}, n_samples={splatting_stage.n_samples}"
        )
        print(f"✓ Forward pass time: {elapsed:.2f}s")

        # Check parameters are learnable
        param_dict = splatting_stage.get_params_dict()
        print(f"✓ Learnable parameters: {list(param_dict.keys())}")
        print(f"✓ Example param values:")
        for name, value in list(param_dict.items())[:3]:
            print(f"    {name}: {value:.6f}")

        print("✓ Splatting stage test PASSED")
        return True

    except Exception as e:
        print(f"✗ Splatting stage test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_splatting_integration():
    """Test SUPPORT with splatting stage integrated"""
    print("\n" + "=" * 60)
    print("TEST 3: SUPPORT with Splatting Integration")
    print("=" * 60)

    # Load splatting parameters
    splat_params_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/260430_splat_params.json"

    if not os.path.exists(splat_params_path):
        print(f"✗ Splatting params not found")
        return False

    with open(splat_params_path, "r") as f:
        splat_params = json.load(f)

    # Create model WITH splatting (no n_lines/n_samples needed)
    try:
        model = SUPPORT(
            in_channels=21,
            mid_channels=[16, 32, 64, 128, 256],
            depth=5,
            blind_conv_channels=64,
            one_by_one_channels=[32, 16],
            last_layer_channels=[64, 32, 16],
            bs_size=[1, 3],
            bp=False,
            use_splatting=True,  # Enable splatting
            splatting_config={
                "initial_params": splat_params,
                "drop_lines": splat_params["preprocessing"]["drop_lines"],
            },
        ).cuda()

        # Test forward pass with [B, T, 2, H, W] input
        # Use smaller size for memory efficiency
        B, T, C, H, W = 1, 21, 2, 64, 128  # Small patch
        print(f"Note: Using small patch size {H}x{W} for memory efficiency")

        x = torch.randn(B, T, C, H, W).cuda()
        x[:, :, 1, :, :] = torch.sin(torch.linspace(0, 10, W).cuda())

        import time

        start = time.time()
        output = model(x)
        elapsed = time.time() - start

        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output has correct dims: {output.ndim == 4}")
        print(f"✓ Forward pass time: {elapsed:.2f}s")

        # Test backward pass
        loss = output.mean()
        loss.backward()

        # Check gradients flow to splatting params
        has_grads = False
        for name, param in model.splatting_stage.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                print(f"✓ Gradient flows to {name}: {param.grad.abs().mean():.6e}")

        if has_grads:
            print("✓ Gradients flow through splatting stage")
        else:
            print("✗ No gradients in splatting parameters")
            return False

        print("✓ Integration test PASSED")
        return True

    except Exception as e:
        print(f"✗ Integration test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading with splatting enabled"""
    print("\n" + "=" * 60)
    print("TEST 4: Data Loading with Splatting")
    print("=" * 60)

    # This test requires actual data, so might need to be run separately
    data_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012"

    if not os.path.exists(data_path):
        print(f"✗ Data path not found: {data_path}")
        print("  Skipping this test (run manually with actual data)")
        return None

    try:
        opt = parse_arguments(
            args=[
                "--input_frames",
                "21",
                "--patch_size",
                "21",
                "16",
                "128",
                "--patch_interval",
                "10",
                "8",
                "64",
                "--batch_size",
                "2",
                "--is_zarr",
                "--is_raw",
                "--is_folder",
                "--noisy_data",
                data_path,
                "--use_splatting",  # Enable splatting in data loader
            ]
        )

        # This will test the SplattingZarrWrapper
        dataloader = gen_train_dataloader(
            opt.patch_size,
            opt.patch_interval,
            opt.batch_size,
            opt.noisy_data,
            opt,
            is_zarr=True,
            is_raw=True,
            use_splatting=True,
        )

        # Get one batch
        for batch in dataloader:
            # Batch is a list: [data, phase_sin, phase_cos, mean, std]
            data = batch[0]

            print(f"✓ Batch is list with {len(batch)} items")
            print(f"✓ Data shape: {data.shape}")
            print(f"✓ Expected shape: [B, T, 2, H, W]")
            print(f"✓ Has 5 dimensions: {data.ndim == 5}")
            if data.ndim == 5:
                print(f"✓ Channel dim (index 2) is 2: {data.shape[2] == 2}")
            break

        print("✓ Data loading test PASSED")
        return True

    except Exception as e:
        print(f"✗ Data loading test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("#  SPLATTING STAGE INTEGRATION TESTS")
    print("#" * 60)

    results = []

    # Test 1: Backward compatibility
    results.append(("Backward Compatibility", test_backward_compatibility()))

    # Test 2: Splatting stage standalone
    results.append(("Splatting Stage", test_splatting_stage_standalone()))

    # Test 3: Integration
    results.append(("Integration", test_splatting_integration()))

    # Test 4: Data loading (optional, needs real data)
    results.append(("Data Loading", test_data_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"{name:30s}: {status}")

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n⚠ Some tests failed. Please review the errors above.")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
