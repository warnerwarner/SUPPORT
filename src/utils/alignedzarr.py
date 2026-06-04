import zarr
import numpy as np


class AlignedZarr:
    def __init__(self, path, shifts):
        self._path = path
        self.shifts = shifts
        self.data = None
        self.pos = None
        self._initialized = False

        with zarr.open(path, mode="r") as store:
            eod_shape = store["eod"].shape
            self.min_length = eod_shape[-1] - np.max(np.abs(self.shifts))
            if self.min_length < 0:
                raise ValueError("Negative min_length, check shift calculation")
            self._shape = (eod_shape[0], eod_shape[1], self.min_length)

    def _lazy_init(self):
        if self._initialized:
            return
        store = zarr.open(self._path, mode="r")
        self.data = store["eod"]
        self.pos = store["position"]
        self._initialized = True

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        self._lazy_init()
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = idx + (slice(None),) * (3 - len(idx))

        t_indices = np.arange(self.shape[0])[idx[0]]
        h_indices = np.arange(self.shape[1])[idx[1]]
        y_indices = np.arange(self.min_length)[idx[2]]

        T, H, Y = np.meshgrid(t_indices, h_indices, y_indices, indexing="ij")

        shifts_for_coords = self.shifts[T, H]

        return self.data.get_coordinate_selection(
            (T, H, shifts_for_coords + Y)
        ).squeeze()

    def _test_get(self, idx):
        self._lazy_init()
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = idx + (slice(None),) * (3 - len(idx))

        t_indices = np.arange(self.shape[0])[idx[0]]
        h_indices = np.arange(self.shape[1])[idx[1]]
        y_indices = np.arange(self.min_length)[idx[2]]

        T, H, Y = np.meshgrid(t_indices, h_indices, y_indices, indexing="ij")

        shifts_for_coords = self.shifts[T, H]

        return self.pos.get_coordinate_selection((T, H, shifts_for_coords + Y))

    def __getattr__(self, name):
        if name in (
            "_path",
            "shifts",
            "data",
            "pos",
            "_initialized",
            "_shape",
            "min_length",
        ):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        self._lazy_init()
        return getattr(self.data, name)

    def __getstate__(self):
        return {
            "_path": self._path,
            "shifts": self.shifts,
            "_shape": self._shape,
            "min_length": self.min_length,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = None
        self.pos = None
        self._initialized = False
