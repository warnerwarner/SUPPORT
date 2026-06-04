import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta"""

    def __init__(self, num_channels, cond_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, num_channels)
        self.beta_proj = nn.Linear(cond_dim, num_channels)

    def forward(self, x, cond):
        # x: (B, C, H, W), cond: (B, cond_dim)
        gamma = self.gamma_proj(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class PhaseEncoder(nn.Module):
    """Encode per-row phase into a conditioning vector per frame."""

    def __init__(self, n_rows=32, embed_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_rows * 2, 64),  # *2 for sin/cos
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, phase_sin, phase_cos):
        # phase_sin, phase_cos: (B, H)
        phase_flat = torch.cat([phase_sin, phase_cos], dim=-1)  # (B, H*2)
        return self.mlp(phase_flat)  # (B, embed_dim)

asasds
