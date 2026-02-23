import torch
import torch.nn as nn
from typing import Optional


class ConvEncoder2D(nn.Module):
    """把 [B, Ny, Nx] 编码到 latent 向量 [B, d]."""
    def __init__(self, nx: int, ny: int, latent_dim: int, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
        )
        self.proj = nn.Linear(channels, latent_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 3:
            x = u.unsqueeze(1)           # [B, 1, Ny, Nx]
        else:
            x = u
        h = self.net(x).flatten(1)  # [B, C]
        z = self.proj(h)             # [B, d]
        return z


class ConvDecoder2D(nn.Module):
    """把 latent 向量 [B, d] 解码回 [B, Ny, Nx]."""
    def __init__(self, nx: int, ny: int, latent_dim: int, channels: int = 64):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.fc = nn.Linear(latent_dim, channels * 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)                       # [B, C*16]
        B = h.shape[0]
        C = h.shape[1] // 16
        h = h.view(B, C, 4, 4)              # [B, C, 4, 4]
        h = torch.nn.functional.interpolate(h, size=(self.ny, self.nx), mode="bilinear", align_corners=False)
        u = self.net(h).squeeze(1)           # [B, Ny, Nx]
        return u


class KoopmanRollout2D(nn.Module):
    """
    2D版本的Koopman潜空间线性演化：
      z_{t+1} = K z_t + B f
    """
    def __init__(self, nx: int, ny: int, latent_dim: int, enc_channels: int = 64, dec_channels: int = 64, use_forcing: bool = True):
        super().__init__()
        self.use_forcing = use_forcing
        self.encoder = ConvEncoder2D(nx, ny, latent_dim, channels=enc_channels)
        self.decoder = ConvDecoder2D(nx, ny, latent_dim, channels=dec_channels)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

        if use_forcing:
            self.force_encoder = ConvEncoder2D(nx, ny, latent_dim, channels=enc_channels)
            self.B = nn.Linear(latent_dim, latent_dim, bias=False)
        else:
            self.force_encoder = None
            self.B = None

    def step(self, u_t: torch.Tensor, F: Optional[torch.Tensor] = None):
        z = self.encoder(u_t)
        if self.use_forcing and F is not None:
            fz = self.force_encoder(F)
            z_next = self.K(z) + self.B(fz)
        else:
            z_next = self.K(z)
        u_next = self.decoder(z_next)
        return u_next, z_next

    def rollout(self, u0: torch.Tensor, steps: int, F: Optional[torch.Tensor] = None):
        """
        u0: [B, Ny, Nx]
        returns:
          u_seq_pred: [B, steps, Ny, Nx] 预测序列
        """
        preds = []
        u = u0
        for _ in range(steps):
            u, _ = self.step(u, F)
            preds.append(u)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return self.encoder(u)
