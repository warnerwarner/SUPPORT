# %%
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# %% # data is inside voltage_imaging/data
output_recon_file = (
    "./data/stephenvoltage_if_61_bs1_wider_1.5x_50epochs_80l1_20l2_composite_mw3.tif"
)
data = tifffile.imread(output_recon_file)
# %%

plt.imshow(data[0], cmap="gray")
# %%
t, h, w = data.shape
denoised_data = data[:, : h // 2, : w // 2]
noised_data = data[:, h // 2 :, : w // 2]
# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im = ax[0].imshow(noised_data[0], cmap="gray")
plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(denoised_data[0], cmap="gray")
plt.colorbar(im, ax=ax[1])
# %%

raw_data_st = ""
data_file = f"../stephen/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00006{raw_data_st}.tif"
_raw_tiff = tifffile.TiffFile(data_file)
raw_data = []
for series in _raw_tiff.series:
    for page in series.pages:
        raw_data.append(page.asarray())
raw_data = np.array(raw_data)

t, x, y = raw_data.shape
raw_data = raw_data.reshape(t // 2, 2, x, y)
raw_data = np.transpose(raw_data, (0, 1, 2, 3))
# %%
normed_raw = (raw_data[:, 0] - np.mean(raw_data[:, 0], axis=0)) / np.std(
    raw_data[:, 0], axis=0
)
# %%
print(normed_raw.shape)
plt.plot(normed_raw.mean(axis=(1, 2)))
# %%
plt.imshow(normed_raw[-1], aspect="auto", interpolation="none")
plt.colorbar()
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pcad = pca.fit_transform(normed_raw.reshape(t // 2, -1))
# %%
print(pcad.shape)
plt.plot(pcad[:, 1])
# %%
plt.imshow(pca.components_[0].reshape(x, y), aspect="auto", interpolation="none")
