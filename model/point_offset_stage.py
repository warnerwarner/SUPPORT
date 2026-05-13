import torch
import torch.nn as nn


class PointOffsetStage(nn.Module):
    """Learn small per-point (dx, dy) corrections before splatting.

    Input:
      value: [B, T, H, W]
      u0:    [B, T, H, W] base x coordinates (splat space)
      v0:    [B, T, H, W] base y coordinates (splat space)
    Output:
      offsets [B, T, 2, H, W] where [:,:,0]=dx and [:,:,1]=dy in splat-pixel units
    """

    def __init__(self, hidden_channels=16, max_offset_px=1.0):
        super().__init__()
        self.max_offset_px = float(max_offset_px)

        # Features per point: [value, u0_norm, v0_norm, t_norm]
        self.net = nn.Sequential(
            nn.Conv3d(4, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, value, u0, v0):
        # value/u0/v0: [B, T, H, W]
        b, t, h, w = value.shape

        device = value.device
        dtype = value.dtype

        t_coord = torch.linspace(-1.0, 1.0, t, device=device, dtype=dtype)
        t_grid = t_coord.view(1, t, 1, 1, 1).expand(b, t, 1, h, w)

        # Normalize base coordinates to roughly [-1, 1]
        u0_min = u0.amin(dim=(1, 2, 3), keepdim=True)
        u0_max = u0.amax(dim=(1, 2, 3), keepdim=True)
        v0_min = v0.amin(dim=(1, 2, 3), keepdim=True)
        v0_max = v0.amax(dim=(1, 2, 3), keepdim=True)
        u0_norm = (u0 - u0_min) / (u0_max - u0_min + 1e-8) * 2.0 - 1.0
        v0_norm = (v0 - v0_min) / (v0_max - v0_min + 1e-8) * 2.0 - 1.0

        feats = torch.cat(
            [value.unsqueeze(2), u0_norm.unsqueeze(2), v0_norm.unsqueeze(2), t_grid],
            dim=2,
        )
        feats = feats.permute(0, 2, 1, 3, 4)  # [B, 4, T, H, W]

        offsets = self.net(feats).permute(0, 2, 1, 3, 4)  # [B, T, 2, H, W]
        offsets = torch.tanh(offsets) * self.max_offset_px
        return offsets
