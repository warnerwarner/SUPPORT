# Notebook checking the output of support on purely modelled data
import numpy as np
import matplotlib.pyplot as plt
import sys

project_root = "/gpfs/data/shohamlab/tom/support"
if project_root not in sys.path:
    sys.path.append(project_root)
from src.train import train, basic_train
from src.utils.util import parse_arguments
from src.utils.dataset import gen_train_dataloader
from model.SUPPORT import SUPPORT
import torch
import numpy as np

import zarr
import numpy as np

sys.path.append("/gpfs/home/warnet02/data/")

import eo_accelerated_2p as eo2p
import importlib
import matplotlib.pyplot as plt

from src.utils.alignedzarr import AlignedZarr
from src.utils.dataset import DatasetSUPPORT_test_stitch
from src.utils.alignment import calculate_peak_shifts
from src.test import validate

# %%

opt = parse_arguments(
    args=[
        "--training_size",
        "5",
        "--noisy_data",
        "/gpfs/data/shohamlab/tom/stephen/zarr/run016",
        "--patch_size",
        "61",
        "100",
        "100",
        "--patch_interval",
        "200",
        "100",
        "100",
        "--input_frames",
        "61",
        "--bs_size",
        "10",
        "10",
        "--depth",
        "5",
        "--is_zarr",
        "--is_folder",
        "--random_seed",
        "22121",
        "--checkpoint_interval",
        "-1",
        "--dataset_key",
        "splatted",
    ]
)

rng = 22121
outs, model = basic_train(opt, rng, epochs=5)
# %%
modlelled_data = np.zeros((2000, 130, 450))
plt.imshow(modlelled_data[1000])


# %%
def basic_train(opt, rng, epochs=5):
    dataloader_train = gen_train_dataloader(
        opt.patch_size,
        opt.patch_interval,
        opt.batch_size,
        opt.noisy_data,  # pass file paths, not pre-loaded tensors
        opt,
        is_zarr=opt.is_zarr,
        is_raw=opt.is_raw,
        rank=0,  # pass rank for distributed cache synchronization
        use_phase_conditioning=opt.use_phase_conditioning,
    )

    model = support(
        in_channels=opt.input_frames,
        mid_channels=opt.unet_channels,
        depth=opt.depth,
        blind_conv_channels=opt.blind_conv_channels,
        one_by_one_channels=opt.one_by_one_channels,
        last_layer_channels=opt.last_layer_channels,
        bs_size=opt.bs_size,
        bp=opt.bp,
    ).cuda()

    optimizer = torch.optim.adam(model.parameters(), lr=opt.lr)
    scaler = torch.cuda.amp.gradscaler(enabled=opt.use_amp)
    rng = np.random.default_rng(rng)

    outs = []
    for epoch in range(epochs):
        outsed = train(
            dataloader_train,
            model,
            optimizer,
            scaler,
            rng,
            none,
            epoch,
            opt,
        )
        print(f"epoch {epoch} loss: {np.mean(outsed[0][-10:])}")
        outs.append(outsed)

    return outs, model
