# Conversion from TIFF to Zarr format

import os
import zarr
import tifffile
import numpy as np
import argparse
import tempfile
from tqdm import tqdm
import dask
import json
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import sys

sys.path.append("/gpfs/home/warnet02/data/")

import eo_accelerated_2p as eo2p


# %%
def tiff_to_zarr(tiff_file, zarr_file, is_raw=True, chunk_size=(61, 16, 320)):
    """Convert a TIFF file to Zarr format."""
    # print(f"Converting {tiff_file} to {zarr_file}...")

    # Read the TIFF file
    frames = []
    with tifffile.TiffFile(tiff_file) as tif:
        for series in tif.series:
            for page in series.pages:
                frames.append(page.asarray())
    frames = np.array(frames)

    if is_raw:
        t, x, y = frames.shape
        frames = frames.reshape(t // 2, 2, x, y)
        # frames = np.transpose(frames, (0, 1, 2, 3))
    mean_val = frames[:, 0, :, :].mean()
    std_val = frames[:, 0, :, :].std()
    zarr_group = zarr.open(zarr_file, mode="w")
    zarr_group.attrs["is_raw"] = is_raw
    eod_group = zarr_group.create_dataset(
        "eod", data=frames[:, 0], chunks=chunk_size, dtype=frames.dtype
    )
    eod_group.attrs["mean"] = mean_val
    eod_group.attrs["std"] = std_val
    pos_group = zarr_group.create_dataset(
        "position", data=frames[:, 1], chunks=chunk_size, dtype=frames.dtype
    )


def build_file_list(base_path):

    file_paths = []

    for root, folders, files in os.walk(base_path):
        for file in files:
            if (
                file.endswith(".tif")
                and "reconstructed" not in file
                and "EOD-on" in file
                and "beads" not in file
            ):
                tiff_path = os.path.join(root, file)
                zarr_path = tiff_path.replace("raw", "zarr").replace(".tif", ".zarr")
                if not os.path.exists(os.path.dirname(zarr_path)):
                    os.makedirs(os.path.dirname(zarr_path))

                file_paths.append((tiff_path, zarr_path))

    return file_paths


def process_one(args):
    tiff_file, zarr_file = args
    return tiff_to_zarr(tiff_file, zarr_file)


# %%
base_path = "/gpfs/home/warnet02/data/stephen/raw"
file_paths = build_file_list(base_path)

# %% using dask

cluster = SLURMCluster(queue="cpu_short", cores=2, memory="32GB", walltime="01:00:00")
cluster.scale(jobs=20)

client = Client(cluster)
# %%
tasks = [dask.delayed(process_one)(pair) for pair in file_paths]
futures = client.compute(tasks)
# %%
print(dask.distributed.as_completed(futures))

# %%
print([f.status for f in futures])
# %%
client.shutdown()


# %%
def recon_to_zarr(zarr_path):
    import sys

    sys.path.append("/gpfs/home/warnet02/data")
    import eo_accelerated_2p as eo2p

    try:
        zarr_in = zarr.open(zarr_path, mode="a")
        zarr_combo = np.array([zarr_in["eod"][:, :, :], zarr_in["position"][:, :, :]])
        zarr_combo = np.transpose(zarr_combo, (1, 0, 2, 3))
        zarr_combo = zarr_combo.reshape(-1, zarr_combo.shape[2], zarr_combo.shape[3])
    except Exception as e:
        print(f"Error in recon_to_zarr: {e}")
        print(f"Error reading zarr file {zarr_path}: {e}")
        return

    print(zarr_combo.shape)

    tiff_path = zarr_path.replace("/zarr", "/raw")
    param_path = None
    for i in os.listdir(os.path.dirname(tiff_path)):
        if i.endswith(".json") and "single" not in i:
            param_path = os.path.join(os.path.dirname(tiff_path), i)
    if param_path is None:
        param_path = "/gpfs/home/warnet02/data/stephen/run012/default_recon_params.json"

    recon_params = json.load(open(param_path, "r"))

    reconed = eo2p.reconstruction.reconstruct(
        zarr_combo, param_path, save_tiff=False, return_binned=True
    )
    print(reconed.shape)

    recon_group = zarr_in.create_dataset(
        "reconstructed",
        data=reconed,
        chunks=(61, reconed.shape[1], reconed.shape[2]),
        dtype=reconed.dtype,
    )
    for key, value in recon_params.items():
        recon_group.attrs[key] = value


# %%
for zarr_path in zarr_paths:
    tiff_path = zarr_path.replace("/zarr", "/raw")
    param_path = None
    for i in os.listdir(os.path.dirname(tiff_path)):
        if i.endswith(".json"):
            param_path = os.path.join(os.path.dirname(tiff_path), i)
    if param_path is None:
        param_path = "/gpfs/home/warnet02/data/stephen/run012/default_recon_params.json"
    print(param_path)
    recon_params = json.load(open(param_path, "r"))
    if recon_params["binning"]["parallelization_mode"] == "multiprocessing":
        recon_params["binning"]["parallelization_mode"] = None
        out_path = param_path.replace(".json", "_singleprocess.json")
        json.dump(recon_params, open(out_path, "w"))
        print(f"new params {out_path}")
# %%
for f in tqdm(zarr_paths):
    recon_to_zarr(f)
# %%
zarr_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run011/stim_v1_fov3_440Hz_10X_2x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00002.zarr"
recon_to_zarr(zarr_path)
# %%
zarr_paths = []
zarr_base = "/gpfs/home/warnet02/data/stephen/zarr"
for root, folders, files in os.walk(zarr_base):
    for folder in folders:
        if folder.endswith(".zarr"):
            # recon_path = os.path.join(root, folder, "reconstructed")
            # if not os.path.exists(recon_path):
            zarr_paths.append(os.path.join(root, folder))
print(len(zarr_paths))
# %%
print(zarr_paths[0])
recon_to_zarr(zarr_paths[0])
# %%

tasks = [dask.delayed(recon_to_zarr)(f) for f in zarr_paths]
futures = client.compute(tasks)

# %%
print(set([f.status for f in futures]))
# %%
for f in futures:
    if f.status == "error":
        print(dir(f))
        print(f.result())
# %%
recon_params = json.load(
    open("/gpfs/home/warnet02/data/stephen/run012/default_recon_params_single.json")
)
print(recon_params["binning"]["parallelization_mode"])


# %%
def add_mean_std(zarr_path):
    zarr_in = zarr.open(zarr_path, mode="a")
    eod_data = zarr_in["reconstructed"][:]
    mean_val = eod_data.mean()
    std_val = eod_data.std()
    zarr_in["reconstructed"].attrs["mean"] = mean_val
    zarr_in["reconstructed"].attrs["std"] = std_val


# %%
for i in tqdm(zarr_paths):
    add_mean_std(i)
