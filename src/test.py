import numpy as np
import torch
import skimage.io as skio
import os
import re

from tqdm import tqdm
from src.utils.dataset import DatasetSUPPORT_test_stitch
from model.SUPPORT import SUPPORT


def validate(test_dataloader, model, use_phase_conditioning=False):
    """
    Validate a model with a test data

    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetSUPPORT_test_stitch!
            Always returns 5 items: (noisy_image, _, single_coordinate, phase_sin, phase_cos)
        model: (Pytorch nn.Module)
        use_phase_conditioning: (bool) DEPRECATED - not used anymore, kept for compatibility

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(
            test_dataloader.dataset.noisy_image.shape, dtype=np.float32
        )

        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, batch_data in enumerate(
            tqdm(test_dataloader, desc="validate")
        ):
            # Always unpack 5 items - phase_sin and phase_cos are always returned
            # but may be empty tensors if not provided
            noisy_image, _, single_coordinate, phase_sin, phase_cos = batch_data

            noisy_image = noisy_image.cuda()  # [b, z, y, x]
            phase_sin = phase_sin.cuda() if phase_sin is not None else None
            phase_cos = phase_cos.cuda() if phase_cos is not None else None

            # Check if phase data is actually present (not empty tensors)
            # Empty tensors have size 1 and were used as placeholders
            if phase_sin is not None and phase_sin.numel() > 1:
                noisy_image_denoised = model(noisy_image, phase_sin, phase_cos).cpu()
            else:
                noisy_image_denoised = model(noisy_image)

            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)):
                stack_start_w = int(single_coordinate["stack_start_w"][bi])
                stack_end_w = int(single_coordinate["stack_end_w"][bi])
                patch_start_w = int(single_coordinate["patch_start_w"][bi])
                patch_end_w = int(single_coordinate["patch_end_w"][bi])

                stack_start_h = int(single_coordinate["stack_start_h"][bi])
                stack_end_h = int(single_coordinate["stack_end_h"][bi])
                patch_start_h = int(single_coordinate["patch_start_h"][bi])
                patch_end_h = int(single_coordinate["patch_end_h"][bi])

                stack_start_s = int(single_coordinate["init_s"][bi])

                denoised_stack[
                    stack_start_s + (T // 2),
                    stack_start_h:stack_end_h,
                    stack_start_w:stack_end_w,
                ] = (
                    noisy_image_denoised[bi]
                    .squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w]
                    .cpu()
                )

        # change nan values to 0 and denormalize
        denoised_stack = (
            denoised_stack * test_dataloader.dataset.std_image.numpy()
            + test_dataloader.dataset.mean_image.numpy()
        )

        return denoised_stack


def getHighestModelFile(model_path):
    pattern = r"model_(\d+)\.pth"
    numbers = []
    for f in os.listdir(model_path):
        match = re.match(pattern, f)
        if match:
            numbers.append(int(match.group(1)))

    max_num = max(numbers)
    model_file = f"{model_path}/model_{max_num}.pth"
    return model_file


if __name__ == "__main__":
    ########## Change it with your data ##############
    data_file = "/gpfs/data/shohamlab/tom/stephen/FOV10_440hz_force1s_9X_1p4x1SAM_0p59t-0p8s-FF_0p25ms-fb_bin1_EOD-on_00001_reconstructed_filtered.tif"
    model_file = getHighestModelFile("./results/saved_models/stephenVoltage_bs10")
    # model_file = "./results/saved_models/stephenVoltage/model_90.pth" # "./results/saved_models/mytest/model_0.pth"
    output_file = "./data/FOV10_440hz_force1s_9X_1p4x1SAM_0p59t-0p8s-FF_0p25ms-fb_bin1_EOD-on_00001_reconstructed_filtered_denoised_bs10.tif"
    patch_size = [61, 64, 64]
    patch_interval = [1, 32, 32]
    batch_size = 16  # lower it if memory exceeds.
    bs_size = 10  # modify if you changed bs_size when training.
    bp_mode = False
    ##################################################

    mod_out = SUPPORT(
        in_channels=61,
        mid_channels=[64, 128, 256, 512, 1024],
        one_by_one_channels=[32, 16],
        blind_conv_channels=64,
        last_layer_channels=[64, 32, 16],
        bs_size=bs_size,
    ).cuda()
    mod_out.out_convs[0] = torch.nn.Conv2d(
        in_channels=128, out_channels=32, kernel_size=1
    ).cuda()
    # model = SUPPORT(in_channels=61, mid_channels=[16, 32, 64, 128, 256], depth=5,\
    # blind_conv_channels=64, one_by_one_channels=[32, 16], last_layer_channels=[64, 32, 16], bs_size=bs_size, bp=bp_mode).cuda()

    # model.load_state_dict(torch.load(model_file))
    print(model_file)
    mod_out.load_state_dict(torch.load(model_file))
    model = mod_out

    demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(
        torch.FloatTensor
    )
    demo_tif = demo_tif[:, :, :]

    testset = DatasetSUPPORT_test_stitch(
        demo_tif, patch_size=patch_size, patch_interval=patch_interval
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    denoised_stack = validate(testloader, model)

    print(denoised_stack.shape)
    skio.imsave(
        output_file,
        denoised_stack[
            (model.in_channels - 1) // 2 : -(model.in_channels - 1) // 2, :, :
        ],
        metadata={"axes": "TYX"},
    )
