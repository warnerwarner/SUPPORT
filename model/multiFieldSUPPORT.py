import torch
import torch.nn as nn
import zarr


# %%


class MultiFieldSUPPORT(nn.Module):
    def __init__(self, hidden=32):
        # self._gen_unet()
        super(MultiFieldSUPPORT, self).__init__()
        self.hidden = hidden
        self._gen_bfnet()
        self.enc_layers = None

    def _gen_unet(self):
        # (Unet) encoding layers
        self.enc_layers = []
        for i in range(len(self.mid_channels)):
            if i == 0:
                self.enc_layers.append(
                    nn.Conv2d(
                        self.in_channels - 1,
                        self.mid_channels[i],
                        kernel_size=3,
                        padding=1,
                    )
                )
            else:
                self.enc_layers.append(
                    nn.Conv2d(
                        self.mid_channels[i - 1],
                        self.mid_channels[i],
                        kernel_size=3,
                        padding=1,
                    )
                )
        self.enc_layers = nn.ModuleList(self.enc_layers)

        # (Unet) decoding layers
        self.dec_layers = []
        for i in range(len(self.mid_channels) - 1):
            self.dec_layers.append(
                nn.Conv2d(
                    self.mid_channels[i] + self.mid_channels[i + 1],
                    self.mid_channels[i],
                    kernel_size=3,
                    padding=1,
                )
            )
        self.dec_layers = nn.ModuleList(reversed(self.dec_layers))

        # (Unet) 1x1 convs
        self.unet_1_convs = []
        for idx, c in enumerate(self.one_by_one_channels):
            if idx == 0:
                self.unet_1_convs.append(
                    nn.Conv2d(self.mid_channels[0], c, kernel_size=1, padding=0)
                )
            else:
                self.unet_1_convs.append(
                    nn.Conv2d(
                        self.one_by_one_channels[idx - 1], c, kernel_size=1, padding=0
                    )
                )
        self.unet_1_convs = nn.ModuleList(self.unet_1_convs)

    def _gen_bfnet(self):
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=self.hidden,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            self.hidden, self.hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        self.conv3 = nn.Conv3d(self.hidden, 1, kernel_size=1)

    def forward_bfnet(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def forward(self, x):
        x = self.forward_bfnet(x)
        return x[:, x.shape[0] // 2]


# %%
zarr_path = "/gpfs/home/warnet02/data/stephen/zarr/12-5-25-force1s-vis-stim/run012/stim_v1_fov4_440Hz_10X_3x1SAM_0p71t-0p9s-FF_0p25ms-fb_bin1_EOD-on_00011.zarr"
zarr_in = zarr.open(zarr_path, mode="r")
recon = zarr_in["reconstructed"][:]
# %%
d = recon[:61]
hidden = d[30].copy()
d = np.concat([d[:30], d[31:]])
d = torch.from_numpy(d).unsqueeze(0).float()
print(d.shape)
# %%
model = MultiFieldSUPPORT()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# %%
for i in range(10):
    video_out = model(d)
    lf = torch.nn.MSELoss()
    loss = lf(video_out, torch.from_numpy(hidden).float().unsqueeze(0))
    print(loss.item())
    loss.backward()
    optimizer.step()
# %%
videt = video_out.squeeze().detach().numpy()
# %%
plt.imshow(videt, aspect="auto", interpolation="none")
# %%
print()
print(video_out.shape)
print(hidden.shape)
# %%

recon.shape
# %%
for root, folders, files in os.walk("/gpfs/home/warnet02/data/stephen/zarr"):
    print(folders)
