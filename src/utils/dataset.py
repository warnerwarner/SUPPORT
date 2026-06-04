import skimage.io as skio
import tifffile
import numpy as np
import torch
import zarr
import pickle
import os
import hashlib
from scipy.signal import hilbert

from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from src.utils.util import get_coordinate
from src.utils.alignment import align_data, calculate_peak_shifts
from src.utils.alignedzarr import AlignedZarr

try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


def _extract_phase_cpu(position_signal):
    """
    Extract phase from sinusoidal position signal using Hilbert transform.

    Args:
        position_signal: (T, H, W) numpy array with sinusoids along W dimension

    Returns:
        phase_sin, phase_cos: (T, H) tensors - sin/cos of phase for continuity
    """

    T, H, W = position_signal.shape

    # Vectorized Hilbert transform along the W dimension (axis=2)
    analytic = hilbert(position_signal, axis=2)

    # Get phase at the center pixel
    phase = np.angle(analytic[:, :, W // 2])

    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)

    return phase_sin, phase_cos


def _extract_phase_gpu(position_signal, batches=10):
    """
    GPU-accelerated version of phase extraction using PyTorch.

    Args:
        position_signal: (T, H, W) numpy array with sinusoids along W dimension
        batches: number of batches to split the data into for GPU processing (int)
    Returns:
        phase_sin, phase_cos: (T, H) tensors - sin/cos of phase for continuity
    """
    phase_sin_list = []
    phase_cos_list = []
    batch_size = position_signal.shape[0] // batches
    if isinstance(position_signal, np.ndarray):
        chunks = np.array_split(position_signal, batches, axis=0)
    elif isinstance(position_signal, torch.Tensor):
        chunks = torch.tensor_split(position_signal, batches, dim=0)
    else:
        raise TypeError(f"Unsupported type: {type(position_signal)}")
    for batch_signal in tqdm(chunks):
        if isinstance(position_signal, np.ndarray):
            batch_signal = torch.from_numpy(batch_signal.astype(np.float32)).cuda()
        elif not batch_signal.is_cuda:
            batch_signal = batch_signal.float().cuda()
        analytic = hilbert_torch(batch_signal, axis=2)
        phase = torch.angle(analytic[:, :, analytic.shape[2] // 2])
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)
        phase_sin_list.append(phase_sin)
        phase_cos_list.append(phase_cos)
    phase_sin = torch.cat(phase_sin_list, dim=0)
    phase_cos = torch.cat(phase_cos_list, dim=0)
    return phase_sin, phase_cos


def extract_phase(position_signal):
    """
    Extract phase from sinusoidal position signal using Hilbert transform.
    Automatically chooses GPU or CPU implementation based on availability.

    Args:
        position_signal: (T, H, W) numpy array with sinusoids along W dimension
    Returns:
        phase_sin, phase_cos: (T, H) tensors - sin/cos of phase for continuity
    """
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        return _extract_phase_gpu(position_signal)
    else:
        return _extract_phase_cpu(position_signal)


def hilbert_torch(x, axis=-1):
    N = x.shape[axis]
    Xf = torch.fft.fft(x, dim=axis)
    h = torch.zeros(N, dtype=Xf.dtype, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[1 : N // 2] = 2
        h[N // 2] = 1
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2
    Xf = Xf * h
    return torch.fft.ifft(Xf, dim=axis)


def random_transform(
    input, target, rng, is_rotate=True, phase_sin=None, phase_cos=None
):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator
        is_rotate: whether to allow 90-degree rotations
        phase_sin: optional phase sin tensor (Pytorch Tensor with dimension [b, T, H])
        phase_cos: optional phase cos tensor (Pytorch Tensor with dimension [b, T, H])

    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
        phase_sin: flipped phase sin tensor if provided, else None
        phase_cos: flipped phase cos tensor if provided, else None
    """
    rand_num = rng.integers(0, 4)  # random number for rotation
    rand_num_2 = rng.integers(0, 2)  # random number for flip

    # Disable rotations when phase conditioning is active — phase is per-row
    # and rotation would mix row/column dimensions, making phase meaningless.
    if is_rotate and phase_sin is None:
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
        # Flip along dim 2 (row axis) of the image
        input = torch.flip(input, dims=[2])
        if target is not None:
            target = torch.flip(target, dims=[2])
        # Flip phase along its row axis (dim 2) to match — phase is (B, T, H)
        if phase_sin is not None:
            phase_sin = torch.flip(phase_sin, dims=[2])
            phase_cos = torch.flip(phase_cos, dims=[2])

    return input, target, phase_sin, phase_cos


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
            print("Alignment applied to raw data using method:", opt.alignment_method)
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
        phase_sin_list=None,
        phase_cos_list=None,
        align_data=False,
        alignment_method="peaks",
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
        self.phase_sin_list = phase_sin_list
        self.phase_cos_list = phase_cos_list
        self.align_data = align_data
        self.alignment_method = alignment_method
        self.alignment_shifts = []

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

            # Handle 4D data for splatting: [T, C, H, W]
            # vs 3D data for normal: [T, H, W]
            if len(tmp_size) == 4:
                # Splatting mode: compare [T, H, W] against patch_size [T, H, W]
                # (skip channel dimension)
                size_to_check = (tmp_size[0], tmp_size[2], tmp_size[3])
            else:
                size_to_check = tmp_size

            if np.any(np.array(size_to_check) < np.array(self.patch_size)):
                raise Exception("patch size is larger than data size")

            # Generate indices for T, H, W dimensions (skipping C if present)
            dims_to_index = [0, 2, 3] if len(tmp_size) == 4 else [0, 1, 2]
            for i, k in enumerate(dims_to_index):
                z_range = list(
                    range(
                        0, tmp_size[k] - self.patch_size[i] + 1, self.patch_interval[i]
                    )
                )
                if tmp_size[k] - self.patch_size[i] > z_range[-1]:
                    z_range.append(tmp_size[k] - self.patch_size[i])
                indices.append(z_range)
            self.indices_ds.append(indices)

        if self.random_patch:
            self.precompute_indices()
            print(
                f"✓ Random patch indices computed for {len(self.precomputed_indices)} patches"
            )

    def precompute_indices(self):
        """
        Precompute random patch indices for each image using vectorized operations.
        This function accounts for images of different sizes by generating random
        indices within the valid range for each image.
        """
        precomputed_indices = []

        # Iterate over each image in the dataset
        for ds_idx, noisy_image in enumerate(self.noisy_images):
            # Get the shape of the image
            shape = noisy_image.shape

            # Determine the number of patches available for this image.
            # Here, we use the precomputed indices list from __init__ (indices_ds)
            # which was generated based on patch_size and patch_interval.
            indices_lists = self.indices_ds[ds_idx]
            count_i = (
                len(indices_lists[0]) * len(indices_lists[1]) * len(indices_lists[2])
            )

            # Handle 4D data for splatting: [T, C, H, W]
            # vs 3D data for normal: [T, H, W]
            if len(shape) == 4:
                # Splatting mode: use T, H, W dimensions (skip C at index 1)
                t_range = shape[0] - self.patch_size[0] + 1
                y_range = shape[2] - self.patch_size[1] + 1
                z_range = shape[3] - self.patch_size[2] + 1
            else:
                # Normal mode: use T, H, W dimensions directly
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
            # Handle 4D (splatting) vs 3D (normal) data
            if len(self.noisy_images[ds_idx].shape) == 4:
                # Splatting: [T, C, H, W] -> slice as [t_range, :, y_range, z_range]
                noisy_image = self.noisy_images[ds_idx][t_range, :, y_range, z_range]
            else:
                noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
        else:
            if self.phase_sin_list is not None:
                # Phase has shape (T_full, H) — slice matching time and row dims
                # Returns (T_patch, H_patch); forward() splits into U-Net and BS-net portions
                phase_sin = self.phase_sin_list[ds_idx][t_range, y_range]
                phase_cos = self.phase_cos_list[ds_idx][t_range, y_range]
                phase_sin = torch.tensor(phase_sin, dtype=torch.float32)
                phase_cos = torch.tensor(phase_cos, dtype=torch.float32)
            else:
                phase_sin = None
                phase_cos = None
            noisy_image_avg = torch.tensor(self.noisy_images[ds_idx].attrs["mean"])
            noisy_image_std = torch.tensor(self.noisy_images[ds_idx].attrs["std"])

            # Handle 4D (splatting) vs 3D (normal) data
            if len(self.noisy_images[ds_idx].shape) == 4:
                # Splatting: [T, C, H, W] -> slice as [t_range, :, y_range, z_range]
                noisy_image = self.noisy_images[ds_idx][t_range, :, y_range, z_range]
            else:
                noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]

            noisy_image = torch.tensor(noisy_image, dtype=torch.float32)
            if self.phase_sin_list is not None:
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
                    phase_sin,
                    phase_cos,
                )

            else:
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
        phase_sin=None,
        phase_cos=None,
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
        self.phase_sin = phase_sin
        self.phase_cos = phase_cos

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
        phase_sin = (
            self.phase_sin[init_s:end_s, init_h:end_h]
            if self.phase_sin is not None
            else torch.empty(1)
        )
        phase_cos = (
            self.phase_cos[init_s:end_s, init_h:end_h]
            if self.phase_cos is not None
            else torch.empty(1)
        )

        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)

        return noisy_image, torch.empty(1), single_coordinate, phase_sin, phase_cos


def gen_train_dataloader(
    patch_size,
    patch_interval,
    batch_size,
    noisy_data_list,
    opt,
    is_zarr=False,
    is_raw=False,
    rank=0,
    use_phase_conditioning=False,
):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data
        opt: options object (must have lazy_loading attribute)
        rank: process rank in distributed training (default 0)
        use_phase_conditioning: whether to extract phase info (bool)

    Returns:
        dataloader_train
    """
    noisy_images_train = []
    phase_sin_list = [] if use_phase_conditioning else None
    phase_cos_list = [] if use_phase_conditioning else None

    for idx, noisy_data in enumerate(noisy_data_list):
        if not is_zarr:
            # Use the helper function to load file
            noisy_image = _load_tif_file(noisy_data, is_raw, opt)
            print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
            noisy_images_train.append(noisy_image)
        else:
            with zarr.open(noisy_data, mode="r") as store:
                if is_raw:
                    noisy_image = store["eod"]

                    if use_phase_conditioning:
                        position_signal = store["position"][:]
                        phase_sin, phase_cos = extract_phase(position_signal)
                        if phase_sin.is_cuda:
                            phase_sin = phase_sin.cpu()
                            phase_cos = phase_cos.cpu()
                        phase_sin_list.append(phase_sin)
                        phase_cos_list.append(phase_cos)
                    if opt.align_data:
                        position_signal = store["position"][:]
                        shifts = calculate_peak_shifts(position_signal)
                        noisy_image = AlignedZarr(noisy_data, shifts=shifts)
                else:
                    noisy_image = store[opt.dataset_key]
            print(
                f"Loaded {noisy_data} Shape : {noisy_image.shape} with key {opt.dataset_key}"
            )
            noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(
        noisy_images_train,
        patch_size=patch_size,
        patch_interval=patch_interval,
        transform=None,
        random_patch=True,
        load_to_memory=not is_zarr,
        phase_sin_list=phase_sin_list if use_phase_conditioning else None,
        phase_cos_list=phase_cos_list if use_phase_conditioning else None,
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
