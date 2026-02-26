import skimage.io as skio
import tifffile
import numpy as np
import torch
import zarr
import pickle
import os
import hashlib

from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from src.utils.util import get_coordinate
from src.utils.alignment import align_data


def random_transform(input, target, rng, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator

    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4)  # random number for rotation
    rand_num_2 = rng.integers(0, 2)  # random number for flip

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=3, dims=(2, 3))

    if rand_num_2 == 1:
        input = torch.flip(input, dims=[2])
        if target is not None:
            target = torch.flip(target, dims=[2])

    return input, target


def normalize(image):
    """
    Normalize the image to [mean/std]=[0/1]

    Arguments:
        image: image stack (Pytorch Tensor with dimension [T, X, Y])

    Returns:
        image: normalized image stack (Pytorch Tensor with dimension [T, X, Y])
        mean_image: mean of the image stack (np.float)
        std_image: standard deviation of the image stack (np.float)
    """
    mean_image = torch.mean(image)
    std_image = torch.std(image)

    image -= mean_image
    image /= std_image

    return image, mean_image, std_image


def _load_tif_file(file_path, is_raw, opt):
    """
    Load a single .tif file and apply preprocessing (is_raw, rolling_mean, etc.)

    Arguments:
        file_path: path to .tif file (str)
        is_raw: whether to apply raw voltage imaging processing (bool)
        opt: options object with align_data, rolling_mean settings

    Returns:
        noisy_image: preprocessed image tensor (torch.FloatTensor with dimension [T, X, Y])
    """
    frames = []
    with tifffile.TiffFile(file_path) as tiff:
        for series in tiff.series:
            for page in series.pages:
                frames.append(page.asarray())

    frames = np.array(frames)

    # Apply is_raw processing (voltage imaging specific)
    if is_raw:
        t, x, y = frames.shape
        frames = frames.reshape(t // 2, 2, x, y)
        frames = np.transpose(frames, (0, 1, 2, 3))
        if opt.align_data:
            frames = align_data(frames, opt.alignment_method, **opt.alignment_kwargs)
        frames = frames[:, 0]
    else:
        if opt.align_data:
            print("Alignment for non-raw data is not possible")

    # Apply rolling mean if specified
    if opt.rolling_mean != 1:
        mean_tiff = np.zeros((frames.shape[0] - opt.rolling_mean, *frames.shape[1:]))
        for frame in range(mean_tiff.shape[0]):
            mean_tiff[frame] = np.mean(frames[frame : frame + opt.rolling_mean], axis=0)
        noisy_image = torch.from_numpy(mean_tiff.astype(np.float32)).type(
            torch.FloatTensor
        )
    else:
        noisy_image = torch.from_numpy(frames.astype(np.float32)).type(
            torch.FloatTensor
        )

    if len(noisy_image.shape) == 2:
        noisy_image = noisy_image.unsqueeze(0)

    return noisy_image


def _get_normalization_cache_path(file_paths, results_dir="./results"):
    """
    Generate a cache file path for mean/std normalization statistics.
    Uses hash of file paths to ensure cache validity.

    Arguments:
        file_paths: list of file paths used in training
        results_dir: directory to store cache file

    Returns:
        cache_path: path to cache file (str)
    """
    # Create hash of all file paths to detect if dataset changes
    paths_str = "|".join(sorted(file_paths))
    paths_hash = hashlib.md5(paths_str.encode()).hexdigest()[:8]

    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"normalization_stats_{paths_hash}.pkl")
    return cache_path


def _load_or_compute_normalization_stats(
    file_paths, is_raw, opt, results_dir="./results", rank=0
):
    """
    Load normalization statistics from cache or compute them if not cached.
    For each file, computes and caches (mean, std) values.

    In distributed training, only rank 0 computes and writes the cache.
    Other ranks wait and then load the cache after rank 0 completes.

    Arguments:
        file_paths: list of file paths (list of str)
        is_raw: whether files are raw voltage imaging data (bool)
        opt: options object
        results_dir: directory to store cache
        rank: process rank in distributed training (default 0)

    Returns:
        mean_std_dict: dictionary mapping file_path -> (mean, std) tuple
    """
    import torch.distributed as dist

    cache_path = _get_normalization_cache_path(file_paths, results_dir)

    # Check if we're in distributed mode
    is_distributed = dist.is_available() and dist.is_initialized()

    # Only rank 0 computes/writes cache in distributed training
    if rank == 0:
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"[Rank 0] Loading normalization statistics from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_dict = pickle.load(f)

            # Check if all files are in cache
            missing_files = [fp for fp in file_paths if fp not in cached_dict]

            if not missing_files:
                print(f"[Rank 0] ✓ All {len(file_paths)} files found in cache")
                mean_std_dict = cached_dict
            else:
                print(
                    f"[Rank 0] ⚠ Cache incomplete: {len(missing_files)} files missing"
                )
                mean_std_dict = cached_dict
        else:
            print(f"[Rank 0] No cache found, will compute normalization statistics")
            mean_std_dict = {}

        # Compute statistics for missing files
        files_to_compute = [fp for fp in file_paths if fp not in mean_std_dict]

        if files_to_compute:
            print(
                f"[Rank 0] Computing normalization statistics for {len(files_to_compute)} files..."
            )
            for file_path in tqdm(files_to_compute, desc="[Rank 0] Computing mean/std"):
                # Load file temporarily
                noisy_image = _load_tif_file(file_path, is_raw, opt)

                # Compute mean and std
                mean_val = torch.mean(noisy_image).item()
                std_val = torch.std(noisy_image).item()
                mean_std_dict[file_path] = (mean_val, std_val)

                # noisy_image goes out of scope and will be garbage collected

            # Save updated cache
            print(f"[Rank 0] Saving normalization statistics to cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(mean_std_dict, f)
            print(f"[Rank 0] ✓ Cache saved with {len(mean_std_dict)} entries")

    # Synchronize: wait for rank 0 to finish writing cache
    if is_distributed:
        if rank == 0:
            print(f"[Rank 0] Broadcasting completion signal to other ranks...")
        dist.barrier()
        if rank != 0:
            print(f"[Rank {rank}] Rank 0 finished, loading cache...")

    # All ranks (including rank 0) load the final cache
    if rank != 0:
        # Non-zero ranks load the cache that rank 0 just created
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                mean_std_dict = pickle.load(f)
            print(f"[Rank {rank}] ✓ Loaded cache with {len(mean_std_dict)} entries")
        else:
            raise FileNotFoundError(
                f"[Rank {rank}] Cache file not found at {cache_path}. "
                f"Rank 0 should have created it!"
            )

    return mean_std_dict


class DatasetSUPPORT_Lazy(Dataset):
    """
    Lazy-loading version of DatasetSUPPORT for .tif files.
    Instead of loading all files into memory, loads files on-demand during training.
    Caches the most recently loaded file per worker to reduce redundant I/O.
    """

    def __init__(
        self,
        file_paths,
        mean_std_dict,
        patch_size=[61, 128, 128],
        patch_interval=[10, 64, 64],
        is_raw=False,
        opt=None,
        transform=None,
        random_patch=True,
        random_patch_seed=0,
    ):
        """
        Arguments:
            file_paths: list of file paths (list of str)
            mean_std_dict: dict mapping file_path -> (mean, std)
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            is_raw: whether files are raw voltage imaging data (bool)
            opt: options object with align_data, rolling_mean, etc.
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
        """
        # Check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.file_paths = file_paths
        self.mean_std_dict = mean_std_dict
        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.is_raw = is_raw
        self.opt = opt
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.precomputed_indices = None

        # Cache for most recently loaded file (per worker process)
        self._cached_file_path = None
        self._cached_file_data = None

        # Pre-compute file shapes and valid patch indices
        # Need to load each file once to get shape
        print(f"Pre-computing patch indices for {len(file_paths)} files...")
        self.file_shapes = []
        self.indices_ds = []
        self.data_weight = []

        for file_path in tqdm(file_paths, desc="Loading file shapes"):
            # Load file to get shape (will be cached for first batch)
            noisy_image = self._load_and_normalize(file_path)
            shape = noisy_image.shape
            self.file_shapes.append(shape)
            self.data_weight.append(np.prod(shape))

            # Compute valid patch indices for this file
            indices = []
            if np.any(np.array(shape) < np.array(self.patch_size)):
                raise Exception(
                    f"Patch size {self.patch_size} is larger than data size {shape} for file {file_path}"
                )

            for k in range(3):
                z_range = list(
                    range(0, shape[k] - self.patch_size[k] + 1, self.patch_interval[k])
                )
                if shape[k] - self.patch_size[k] > z_range[-1]:
                    z_range.append(shape[k] - self.patch_size[k])
                indices.append(z_range)
            self.indices_ds.append(indices)

        print(f"✓ Patch indices computed for {len(file_paths)} files")
        if self.random_patch:
            self.precompute_indices()
            print(
                f"✓ Random patch indices computed for {len(self.precomputed_indices)} patches"
            )

        means = [self.mean_std_dict[fp][0] for fp in self.file_paths]
        stds = [self.mean_std_dict[fp][1] for fp in self.file_paths]
        self.mean_images = torch.tensor(means)
        self.std_images = torch.tensor(stds)

    def _load_and_normalize(self, file_path):
        """Load a single file and apply normalization using cached mean/std"""
        # Load file
        noisy_image = _load_tif_file(file_path, self.is_raw, self.opt)

        # Apply normalization using cached statistics
        mean_val, std_val = self.mean_std_dict[file_path]
        noisy_image -= mean_val
        noisy_image /= std_val

        return noisy_image

    def precompute_indices(self):
        """
        Precompute random patch indices for each file.
        Called once per epoch to generate new random patches.
        """
        precomputed_indices = []

        for ds_idx, file_path in enumerate(self.file_paths):
            shape = self.file_shapes[ds_idx]
            indices_lists = self.indices_ds[ds_idx]
            count_i = (
                len(indices_lists[0]) * len(indices_lists[1]) * len(indices_lists[2])
            )

            # Calculate valid range for each dimension
            t_range = shape[0] - self.patch_size[0] + 1
            y_range = shape[1] - self.patch_size[1] + 1
            z_range = shape[2] - self.patch_size[2] + 1

            # Generate random indices
            t_indices = self.patch_rng.integers(0, t_range, size=count_i)
            y_indices = self.patch_rng.integers(0, y_range, size=count_i)
            z_indices = self.patch_rng.integers(0, z_range, size=count_i)

            indices_for_file = [
                (ds_idx, int(t), int(y), int(z))
                for t, y, z in zip(t_indices, y_indices, z_indices)
            ]
            precomputed_indices.extend(indices_for_file)

        # Shuffle to randomize order
        self.patch_rng.shuffle(precomputed_indices)
        self.precomputed_indices = precomputed_indices

    def __len__(self):
        total = 0
        for indices in self.indices_ds:
            total += len(indices[0]) * len(indices[1]) * len(indices[2])
        return total

    def __getitem__(self, i):
        """Load patch on-demand from disk"""
        # Get patch location
        if self.random_patch:
            ds_idx, t_idx, y_idx, z_idx = self.precomputed_indices[i]
        else:
            ds_idx = 0
            t_idx = self.indices_ds[ds_idx][0][
                i // (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))
            ]
            y_idx = self.indices_ds[ds_idx][1][
                (
                    i
                    % (
                        len(self.indices_ds[ds_idx][1])
                        * len(self.indices_ds[ds_idx][2])
                    )
                )
                // len(self.indices_ds[ds_idx][2])
            ]
            z_idx = self.indices_ds[ds_idx][2][i % len(self.indices_ds[ds_idx][2])]

        file_path = self.file_paths[ds_idx]

        # Load file if not cached (cache per worker process)
        if file_path != self._cached_file_path:
            self._cached_file_data = self._load_and_normalize(file_path)
            self._cached_file_path = file_path

        # Extract patch from cached data
        t_range = slice(t_idx, t_idx + self.patch_size[0])
        y_range = slice(y_idx, y_idx + self.patch_size[1])
        z_range = slice(z_idx, z_idx + self.patch_size[2])

        noisy_image = self._cached_file_data[t_range, y_range, z_range]

        return (
            noisy_image,
            torch.tensor(
                [
                    [t_idx, t_idx + self.patch_size[0]],
                    [y_idx, y_idx + self.patch_size[1]],
                    [z_idx, z_idx + self.patch_size[2]],
                ]
            ),
            torch.tensor(ds_idx),
        )


class DatasetSUPPORT(Dataset):
    def __init__(
        self,
        noisy_images,
        patch_size=[61, 128, 128],
        patch_interval=[10, 64, 64],
        load_to_memory=True,
        transform=None,
        random_patch=True,
        random_patch_seed=0,
    ):
        """
        Arguments:
            noisy_images: list of noisy image stack ([Tensor with dimension [t, x, y]])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            algorithm: the algorithm of use (str)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        # initialize
        self.data_weight = []
        for noisy_image in noisy_images:
            if load_to_memory:
                self.data_weight.append(torch.numel(noisy_image))
            else:
                self.data_weight.append(np.prod(noisy_image.shape))

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.precomputed_indices = None
        self.load_to_memory = load_to_memory

        self.noisy_images = noisy_images
        self.mean_images = []
        self.std_images = []
        if load_to_memory:
            for idx, noisy_image in enumerate(tqdm(noisy_images)):
                noisy_image, mean_image, std_image = normalize(noisy_image)
                self.noisy_images[idx] = noisy_image
                self.mean_images.append(mean_image)
                self.std_images.append(std_image)
            self.mean_images = torch.tensor(self.mean_images)
            self.std_images = torch.tensor(self.std_images)
            print("Normalized noisy images and stored mean/std in memory.")
            print(f"Mean: {self.mean_images}, Std: {self.std_images}")

        # generate index
        self.indices_ds = []
        for noisy_image in self.noisy_images:
            indices = []
            tmp_size = noisy_image.shape
            if np.any(tmp_size < np.array(self.patch_size)):
                raise Exception("patch size is larger than data size")

            for k in range(3):
                z_range = list(
                    range(
                        0, tmp_size[k] - self.patch_size[k] + 1, self.patch_interval[k]
                    )
                )
                if tmp_size[k] - self.patch_size[k] > z_range[-1]:
                    z_range.append(tmp_size[k] - self.patch_size[k])
                indices.append(z_range)
            self.indices_ds.append(indices)

    def precompute_indices(self):
        """
        Precompute random patch indices for each image using vectorized operations.
        This function accounts for images of different sizes by generating random
        indices within the valid range for each image.
        """
        precomputed_indices = []

        # Iterate over each image in the dataset
        for ds_idx, noisy_image in enumerate(self.noisy_images):
            # Get the shape of the image (T, H, W)
            shape = noisy_image.shape

            # Determine the number of patches available for this image.
            # Here, we use the precomputed indices list from __init__ (indices_ds)
            # which was generated based on patch_size and patch_interval.
            indices_lists = self.indices_ds[ds_idx]
            count_i = (
                len(indices_lists[0]) * len(indices_lists[1]) * len(indices_lists[2])
            )

            # Calculate the valid range for each dimension
            t_range = shape[0] - self.patch_size[0] + 1
            y_range = shape[1] - self.patch_size[1] + 1
            z_range = shape[2] - self.patch_size[2] + 1

            # Generate random indices in a vectorized way for the current image
            t_indices = self.patch_rng.integers(0, t_range, size=count_i)
            y_indices = self.patch_rng.integers(0, y_range, size=count_i)
            z_indices = self.patch_rng.integers(0, z_range, size=count_i)

            # Create a list of tuples (ds_idx, t_idx, y_idx, z_idx) for this image
            indices_for_image = [
                (ds_idx, int(t), int(y), int(z))
                for t, y, z in zip(t_indices, y_indices, z_indices)
            ]
            precomputed_indices.extend(indices_for_image)

        # Shuffle the complete list of precomputed indices to randomize order per epoch
        self.patch_rng.shuffle(precomputed_indices)
        self.precomputed_indices = precomputed_indices

    def __len__(self):
        total = 0
        for indices in self.indices_ds:
            total += len(indices[0]) * len(indices[1]) * len(indices[2])

        return total

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            ds_idx, t_idx, y_idx, z_idx = self.precomputed_indices[i]
        else:
            ds_idx = 0
            t_idx = self.indices_ds[ds_idx][0][
                i // (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))
            ]
            y_idx = self.indices_ds[ds_idx][1][
                (
                    i
                    % (
                        len(self.indices_ds[ds_idx][1])
                        * len(self.indices_ds[ds_idx][2])
                    )
                )
                // len(self.indices_ds[ds_idx][2])
            ]
            z_idx = self.indices_ds[ds_idx][2][i % len(self.indices_ds[ds_idx][2])]

        # input dataset range
        t_range = slice(t_idx, t_idx + self.patch_size[0])
        y_range = slice(y_idx, y_idx + self.patch_size[1])
        z_range = slice(z_idx, z_idx + self.patch_size[2])

        if self.load_to_memory:
            noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
        else:
            noisy_image_avg = torch.tensor(self.noisy_images[ds_idx].attrs["mean"])
            noisy_image_std = torch.tensor(self.noisy_images[ds_idx].attrs["std"])
            noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
            noisy_image = torch.tensor(noisy_image, dtype=torch.float32)
            return (
                noisy_image,
                torch.tensor(
                    [
                        [t_idx, t_idx + self.patch_size[0]],
                        [y_idx, y_idx + self.patch_size[1]],
                        [z_idx, z_idx + self.patch_size[2]],
                    ]
                ),
                torch.tensor(ds_idx),
                noisy_image_avg,
                noisy_image_std,
            )

        return (
            noisy_image,
            torch.tensor(
                [
                    [t_idx, t_idx + self.patch_size[0]],
                    [y_idx, y_idx + self.patch_size[1]],
                    [z_idx, z_idx + self.patch_size[2]],
                ]
            ),
            torch.tensor(ds_idx),
        )


class DatasetSUPPORT_test_stitch(Dataset):
    def __init__(
        self,
        noisy_image,
        patch_size=[61, 128, 128],
        patch_interval=[10, 64, 64],
        load_to_memory=True,
        transform=None,
        random_patch=False,
        random_patch_seed=0,
    ):
        """
        Arguments:
            noisy_image: noisy image stack (Tensor with dimension [t, x, y])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.noisy_image = noisy_image
        self.noisy_image, self.mean_image, self.std_image = normalize(self.noisy_image)

        # generate index
        self.indices = []
        tmp_size = self.noisy_image.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)

    def __len__(self):
        return len(
            self.indices
        )  # len(self.indices[0]) * len(self.indices[1]) * len(self.indices[2])

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]

        # input dataset range
        init_h = single_coordinate["init_h"]
        end_h = single_coordinate["end_h"]
        init_w = single_coordinate["init_w"]
        end_w = single_coordinate["end_w"]
        init_s = single_coordinate["init_s"]
        end_s = single_coordinate["end_s"]

        # for stitching dataset range
        noisy_image = self.noisy_image[init_s:end_s, init_h:end_h, init_w:end_w]

        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)

        return noisy_image, torch.empty(1), single_coordinate


def gen_train_dataloader(
    patch_size,
    patch_interval,
    batch_size,
    noisy_data_list,
    opt,
    is_zarr=False,
    is_raw=False,
    rank=0,
):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data
        opt: options object (must have lazy_loading attribute)
        rank: process rank in distributed training (default 0)

    Returns:
        dataloader_train
    """
    # Check if lazy loading is enabled (only for .tif files, not zarr)
    use_lazy_loading = hasattr(opt, "lazy_loading") and opt.lazy_loading and not is_zarr

    if use_lazy_loading:
        print("=" * 70)
        print("LAZY LOADING MODE ENABLED")
        print("=" * 70)
        print(f"Files will be streamed from disk instead of loaded into RAM")
        print(
            f"This allows training with many more files (current: {len(noisy_data_list)} files)"
        )
        print("=" * 70)

        # Compute or load normalization statistics from cache
        # Only rank 0 computes, others wait and load
        mean_std_dict = _load_or_compute_normalization_stats(
            noisy_data_list, is_raw, opt, results_dir=opt.results_dir, rank=rank
        )

        # Create lazy-loading dataset
        dataset_train = DatasetSUPPORT_Lazy(
            file_paths=noisy_data_list,
            mean_std_dict=mean_std_dict,
            patch_size=patch_size,
            patch_interval=patch_interval,
            is_raw=is_raw,
            opt=opt,
            transform=None,
            random_patch=True,
        )

        print("=" * 70)
        print(f"✓ Lazy loading dataset created with {len(noisy_data_list)} files")
        print(f"  Memory usage: Minimal (~2-3 GB per node)")
        print(f"  Dataset size: {len(dataset_train):,} patches")
        print("=" * 70)

    else:
        # Original eager-loading behavior
        noisy_images_train = []

        for noisy_data in noisy_data_list:
            if not is_zarr:
                # Use the helper function to load file
                noisy_image = _load_tif_file(noisy_data, is_raw, opt)
                print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
                noisy_images_train.append(noisy_image)
            else:
                zarr_data = zarr.open(noisy_data, mode="r")
                if is_raw:
                    noisy_image = zarr_data["eod"]
                else:
                    noisy_image = zarr_data["reconstructed"]
                print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
                noisy_images_train.append(noisy_image)

        dataset_train = DatasetSUPPORT(
            noisy_images_train,
            patch_size=patch_size,
            patch_interval=patch_interval,
            transform=None,
            random_patch=True,
            load_to_memory=not is_zarr,
        )

    # Create DataLoader (same for both lazy and eager loading)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        prefetch_factor=opt.prefetch_factor,
    )

    return dataloader_train
