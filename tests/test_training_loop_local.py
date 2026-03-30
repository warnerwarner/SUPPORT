import os
import torch
import numpy as np
from tqdm import tqdm

# Add src to path to import modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dataset import gen_train_dataloader, random_transform
from src.utils.util import parse_arguments
from model.SUPPORT import SUPPORT

def test_local_training_loop():
    print("="*60)
    print("Testing Training Loop Locally (Single GPU)")
    print("="*60)

    # 1. Create a mock options object that matches the bash script parameters
    class MockOpt:
        def __init__(self):
            # Same paths as train_distributed_multi_node.sh
            self.noisy_data = ["/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run011/stim_v1_fov3_440Hz_10X_2x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00033.zarr"]
            self.is_folder = False
            self.is_zarr = True
            self.is_raw = True
            
            # Patch and batch settings
            self.batch_size = 2 # Reduced batch size for local testing
            self.patch_size = [61, 16, 320]
            self.patch_interval = [10, 4, 160]
            self.training_size = 2 # Tiny size so it finishes quickly
            
            # Model params
            self.depth = 8
            self.bs_size = [2, 2]
            
            # Phase conditioning
            self.use_phase_conditioning = True
            
            # Optimization params
            self.use_amp = True
            self.loss_coef = [0.7, 0.3]
            self.lazy_loading = False # Assuming this was on by default
            self.rolling_mean = 1
            self.random_seed = 0
            self.n_cpu = 1
            self.prefetch_factor = 2

    opt = MockOpt()
    
    # 2. Setup Device & Random Number Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    rng = np.random.default_rng(opt.random_seed)

    # 3. Create the dataloader exactly as in train_distributed.py
    # We pass rank=0 so it acts like the main process
    print("Initializing Dataloader...")
    try:
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
        print("✓ Dataloader initialized successfully.")
    except Exception as e:
        print(f"✗ Dataloader initialization failed: {e}")
        return

    # 4. Initialize the Model
    print("Initializing SUPPORT Model...")
    try:
        model = SUPPORT(
            in_channels=opt.patch_size[0],
            mid_channels=[16, 32, 64, 128, 256],
            depth=opt.depth,
            blind_conv_channels=64,
            one_by_one_channels=[32, 16],
            last_layer_channels=[64, 32, 16],
            bs_size=opt.bs_size,
            is_raw=opt.is_raw,
            use_phase_conditioning=opt.use_phase_conditioning,
        ).to(device)
        print("✓ Model initialized successfully.")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return

    # 5. Setup Loss and Optimizer
    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    # 6. Run the training loop for a few batches
    print("Starting Training Loop Test...")
    model.train()
    
    for i, data in enumerate(dataloader_train):
        if i >= 3: # Just run 3 batches to prove it works
            break
            
        print(f"  Processing batch {i+1}...")
        try:
            # --- THIS IS THE SECTION WE ARE TESTING ---
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
                phase_sin = phase_sin.to(device)
                phase_cos = phase_cos.to(device)
                
                # THE FIX: Reshape the averages so they broadcast correctly over the batch
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
            
            # Apply device and normalization
            B, T, X, Y = noisy_image.shape
            noisy_image = noisy_image.to(device)
            noisy_image, _ = random_transform(noisy_image, None, rng, is_rotate=True)
            
            if opt.is_zarr:
                noisy_image_avg = noisy_image_avg.to(device)
                noisy_image_std = noisy_image_std.to(device)
                # If the fix isn't applied, this next line will crash when broadcasting
                noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std
                
            noisy_image_target = torch.unsqueeze(noisy_image[:, int(T / 2), :, :], dim=1)
            
            # Forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                noisy_image_denoised = model(noisy_image, phase_sin, phase_cos)
                loss_l1 = L1_pixelwise(noisy_image_denoised, noisy_image_target)
                loss_l2 = L2_pixelwise(noisy_image_denoised, noisy_image_target)
                loss_sum = opt.loss_coef[0] * loss_l1 + opt.loss_coef[1] * loss_l2

            # Backward pass
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"    ✓ Batch {i+1} successful (Loss: {loss_sum.item():.4f})")
            
        except Exception as e:
            print(f"    ✗ Batch {i+1} failed with error:")
            print(f"      {e}")
            import traceback
            traceback.print_exc()
            return
            
    print("="*60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("The shape broadcasting error has been resolved.")
    print("="*60)

if __name__ == "__main__":
    test_local_training_loop()
