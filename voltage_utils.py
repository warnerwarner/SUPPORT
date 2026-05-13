import zarr
import numpy as np
import tifffile
import os
import json
from tqdm import tqdm
import logging
import re
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_tif_to_zarr(
    tif_path: str, zarr_path: str, type: str = "raw", overwrite: bool = False, **kwargs
):
    """convert tiff file to zarr format for use with support

    Parameters
    ----------
    tif_path : str
        path to the tiff file to convert
    zarr_path : str
        output path for the zarr file
    type : str
        type of data the tif contains, can be raw, splatted, or
        splatted_aligned_to_session
    overwrite : bool
        whether to overwrite existing zarr file. If True, existing zarr file will be deleted before conversion. If False, data will not be converted if the file already exists.
    """

    assert type in [
        "eod",
        "splatted",
        "splatted_aligned_to_session",
    ], "Type must be either 'eod' or 'splatted' or splatted_aligned_to_session'."

    if os.path.exists(zarr_path) and not overwrite:
        logging.info(f"Zarr file {zarr_path} already exists. Appending data.")
        zarr_data = zarr.open(zarr_path, mode="a")
    else:
        logging.info(
            f"Creating new Zarr file at {zarr_path}. Overwrite is set to {overwrite}."
        )
        zarr_data = zarr.open(zarr_path, mode="w")

    if overwrite and type in zarr_data:
        logging.info(
            f"Dataset {type} already exists in Zarr file. Overwriting dataset."
        )
        del zarr_data[type]
    elif type in zarr_data:
        logging.info(f"Dataset {type} already exists in Zarr file")
        return
    # Load the TIFF file
    tiff_data = []
    for series in tifffile.TiffFile(tif_path).series:
        for page in series.pages:
            tiff_data.append(page.asarray())

    tiff_data = np.array(tiff_data)
    logging.info(f"Loaded TIFF data with shape {tiff_data.shape} from {tif_path}")
    if type == "eod":
        logging.info("Extracting position data from eod TIFF.")
        pos = tiff_data[1::2]
        tiff_data = tiff_data[::2]

    mean_data = np.mean(tiff_data)
    std_data = np.std(tiff_data)
    zarr_data.create_dataset(type, data=tiff_data)
    zarr_data[type].attrs["mean"] = mean_data
    zarr_data[type].attrs["std"] = std_data

    if type == "eod":
        logging.info("Saving position data to Zarr.")
        zarr_data.create_dataset("position", data=pos)

    if kwargs:
        for key, value in kwargs.items():
            zarr_data[type].attrs[key] = value
            logging.info(f"Added attribute {key} to Zarr dataset with value {value}")


def convert_directory(
    dir_path: str,
    suffixes_keys: List[str],
    zarr_dir: str,
    pattern: str = "\d{5}",
    suffix_kwargs: dict = None,
    overwrite: bool = False,
):
    """convert all tif files matching a pattern to zarr format

    Parameters
    ----------
    dir_path : str
        path to the directory
    suffixes_keys : List[str]
        keys to indicate the type of data the tif contains. Matches to
        strings between the file number and .tif
    zarr_dir : str
        directory to save the zarr data to
    pattern : str
        pattern to split data between file name and the
        suffixes. Typically will be a series of numbers
    suffix_kwargs : dict
        dictionary of additional keyword arguments to pass to the convert_tif_to_zarr function for each suffix key.
    overwrite : bool
        whether to overwrite existing zarr files or append to them. If True, existing zarr files will be deleted before conversion. If False, data will not be converted if the file already exists.

    """

    pattern = re.compile(pattern)
    zarr_paths = []
    tif_paths = []
    suffixes = []
    for file in os.listdir(dir_path):
        if file.endswith(".tif"):
            result = pattern.search(file)
            if result:
                suffix = file.split(result.group())[1][1:-4]
                if suffix == "":
                    suffix = "eod"
                main_name = file.split(result.group())[0] + result.group() + ".zarr"

                if suffix in suffixes_keys:

                    logging.debug(
                        f"Processing file: {file} with suffix: {suffix} and main name: {main_name}"
                    )
                    zarr_path = os.path.join(zarr_dir, main_name)
                    tif_path = os.path.join(dir_path, file)
                    zarr_paths.append(zarr_path)
                    tif_paths.append(tif_path)
                    suffixes.append(suffix)

    unique_suffixes = np.unique(suffixes)
    logging.info(
        f"Found {len(tif_paths)} tiff files containing {len(unique_suffixes)} ({[i for i in unique_suffixes]}) suffixes, to be converted to {len(np.unique(zarr_paths))} zarr files."
    )
    for tif_path, zarr_path, suffix in zip(tqdm(tif_paths), zarr_paths, suffixes):
        logging.info(f"Converting {tif_path} to {zarr_path} with suffix {suffix}.")
        try:
            if suffix_kwargs and suffix in suffix_kwargs:
                convert_tif_to_zarr(
                    tif_path,
                    zarr_path,
                    type=suffix,
                    overwrite=overwrite,
                    **suffix_kwargs[suffix],
                )
            else:
                convert_tif_to_zarr(
                    tif_path, zarr_path, type=suffix, overwrite=overwrite
                )
        except Exception as e:
            logging.error(f"Error converting {tif_path} to {zarr_path}: {e}")


def add_attributes_to_zarr(zarr_path: str, dataset: str, attributes: dict):
    """add attributes to an existing zarr file

    Parameters
    ----------
    zarr_path : str
        path to the zarr file to add attributes to
    dataset: str
        name of the dataset to add attributes to.
    attributes : dict
        dictionary of attributes to add to the zarr file. Keys are the attribute names and values are the attribute values.
    """
    if not os.path.exists(zarr_path):
        logging.error(f"Zarr file {zarr_path} does not exist. Cannot add attributes.")
        return

    zarr_data = zarr.open(zarr_path, mode="a")
    for key, value in attributes.items():
        zarr_data[dataset].attrs[key] = value
        logging.info(f"Added attribute {key} to Zarr file with value {value}")


# %%
if __name__ == "__main__":
    splat_params = json.load(
        open("/gpfs/home/warnet02/data/stephen/run016/splat_params.json", "r")
    )

    logger = logging.getLogger()
    dir_path = "/gpfs/home/warnet02/data/stephen/run016"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    suffixes_keys = ["eod", "splatted", "splatted_aligned_to_session"]
    zarr_dir = "/gpfs/home/warnet02/data/stephen/zarr/run016"
    kwargs_dict = {
        "splatted": {"splat_params": splat_params},
        "splatted_aligned_to_session": {"splat_params": splat_params},
    }
    convert_directory(
        dir_path, suffixes_keys, zarr_dir, pattern="\d{5}", suffix_kwargs=kwargs_dict
    )
