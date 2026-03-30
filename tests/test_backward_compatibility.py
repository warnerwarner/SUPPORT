import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.SUPPORT import SUPPORT


def test_no_phase_mode():
    print("Testing model with phase conditioning DISABLED...")
    n_frames = 61
    patch_h, patch_w = 16, 320

    # Initialize model without phase encoding
    model = SUPPORT(
        in_channels=n_frames,
        use_phase_conditioning=False,
        mid_channels=[16, 32],
        one_by_one_channels=[16, 8],
        blind_conv_channels=16,
        last_layer_channels=[8, 4],
    )

    # Check if BSNet input channels are correct (should be just unet_out channels = 8)
    # model.one_by_one_channels[-1] is 8
    bs_in_channels = model.conv3x3[0].in_channels
    print(f"BSNet input channels: {bs_in_channels}")
    assert bs_in_channels == 8, f"Expected 8 channels, got {bs_in_channels}"

    # Dummy data
    x = torch.randn(1, n_frames, patch_h, patch_w)

    # Test forward pass with None inputs
    print("Forward pass with phase=None...")
    try:
        output = model(x, phase_sin=None, phase_cos=None)
        assert output.shape == (1, 1, patch_h, patch_w)
        print("✓ Forward pass successful with None.")
    except Exception as e:
        print(f"✗ Forward pass failed with None: {e}")
        raise e

    # Test forward pass with dummy tensors (should still ignore them because use_phase_conditioning is False)
    print("Forward pass with dummy phase tensors...")
    try:
        p_sin = torch.randn(1, n_frames, patch_h)
        p_cos = torch.randn(1, n_frames, patch_h)
        output = model(x, p_sin, p_cos)
        assert output.shape == (1, 1, patch_h, patch_w)
        print("✓ Forward pass successful.")
    except Exception as e:
        print(f"✗ Forward pass failed with dummy tensors: {e}")
        raise e

    print("-" * 30)
    print("Backward compatibility test PASSED!")


if __name__ == "__main__":
    try:
        test_no_phase_mode()
    except Exception:
        sys.exit(1)
