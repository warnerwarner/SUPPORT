import tifffile
from scipy.signal import find_peaks, correlate
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import time


def _align_data_correlation(
    data: np.ndarray, reference: str = "first", cut_off: int = 100
) -> np.ndarray[np.float32, 4]:
    """Align data using cross-correlation to a reference line

    Parameters
    ----------
    data : np.ndarray
        input data, of shape frames x 2 x lines x time
    reference : str
        what to use as the reference, defaults to 'first' - which uses
        the first line in the dataset to align to
    cut_off : int
        the number of samples in the reference to use for alignment,
        defaults to 100.

    Returns
    -------
    ndarray[np.float32, 4]
        Aligned data, both EOD position and detector values have been
        aligned per row and per frame

    Raises
    ------
    ValueError
        If the reference is not a known option

    """

    # Assume data shape: (frames, 2, rows, samples)
    frames, chans, rows, samples = data.shape

    # Preprocess: zero negatives
    data[:, 1, :, :][data[:, 1, :, :] < 0] = 0

    # Reference line (broadcasted for all rows)
    if reference == "first":
        reference_line = data[0, 1, 0, :cut_off]
        reference_line = reference_line - np.mean(reference_line)
    else:
        raise ValueError(
            "Unknown reference type, please use first"
        )  # todo implement other references maybe

    # Prepare output arrays
    aligned_data = np.zeros_like(data)
    lengths = np.zeros((frames, rows), dtype=np.int32)
    shifts = np.zeros((frames, rows), dtype=np.int32)

    for frame_idx in range(frames):
        lines = data[frame_idx, 1, :, :cut_off]
        lines = lines - np.mean(lines, axis=1, keepdims=True)

        # Compute correlations (vectorised over rows)
        # Output shape: (rows, cut_off*2 -1)
        correlations = np.array(
            [correlate(reference_line, line, mode="full") for line in lines]
        )

        # Find shifts (vectorised)
        shift = correlations.argmax(axis=1) - (lines.shape[1] - 1)
        shifts[frame_idx, :] = shift

        # Now align all rows in this frame according to shift
        for row_idx in range(rows):
            s = shift[row_idx]
            if s < 0:
                aligned_data[frame_idx, :, row_idx, :s] = data[
                    frame_idx, :, row_idx, -s:
                ]
                lengths[frame_idx, row_idx] = samples + s
            elif s > 0:
                aligned_data[frame_idx, :, row_idx, s:] = data[
                    frame_idx, :, row_idx, :-s
                ]
                lengths[frame_idx, row_idx] = samples - s
            else:
                aligned_data[frame_idx, :, row_idx, :] = data[frame_idx, :, row_idx, :]
                lengths[frame_idx, row_idx] = samples

    aligned_data = aligned_data[:, :, :, : np.min(lengths)]
    return aligned_data


def _align_data_peaks(
    data: np.ndarray, height: int = 1000
) -> np.ndarray[np.float32, 4]:
    """Align data using peak finding to a reference line

    Parameters
    ----------
    data : np.ndarray
        input data, of shape frames x 2 x lines x time
    height : int
        the minimum height of peaks to consider, defaults to 1000

    Returns
    -------
    ndarray[np.float32, 4]
        Aligned data, both EOD position and detector values have been
        aligned per row and per frame

    """
    data[:, 1, :, :][data[:, 1, :, :] < 0] = 0
    aligned_data = np.zeros_like(data)
    lengths = np.zeros((data.shape[0] * data.shape[2],), dtype=np.int32)
    count = 0
    for frame_idx in trange(data.shape[0]):
        for row_idx in range(data.shape[2]):
            line = data[frame_idx, 1, row_idx, :100]
            peaks, _ = find_peaks(line, height=height)
            algined_line = line[peaks[0] :]
            aligned_data[frame_idx, 0, row_idx, : -peaks[0]] = data[
                frame_idx, 0, row_idx, peaks[0] :
            ]
            aligned_data[frame_idx, 1, row_idx, : -peaks[0]] = data[
                frame_idx, 1, row_idx, peaks[0] :
            ]

            lengths[count] = data.shape[-1] - peaks[0]
            count += 1

    aligned_data = aligned_data[:, :, :, : np.min(lengths)]
    return aligned_data


def align_data(
    data, method: str = "correlation", **kwargs
) -> np.ndarray[np.float32, 4]:
    """Align data using the specified method

    Parameters
    ----------
    data : np.ndarray
        input data, of shape frames x 2 x lines x time
    method : str
        the method to use for alignment, either 'correlation' or 'peaks',
        defaults to 'correlation'
    **kwargs
        additional keyword arguments to pass to the alignment function

    Returns
    -------
    ndarray[np.float32, 4]
        Aligned data, both EOD position and detector values have been
        aligned per row and per frame
    Raises
    ------
    ValueError
        If the method is not a known option
    """
    if method == "correlation":
        func = _align_data_correlation
    elif method == "peaks":
        func = _align_data_peaks
    else:
        raise ValueError(
            "Unknown alignment method, please use 'correlation' or 'peaks'"
        )
    aligned_data = func(data, **kwargs)
    print(
        f"Data was aligned using {method}, original shape: {data.shape}, new shape:{aligned_data.shape}"
    )
    return aligned_data
