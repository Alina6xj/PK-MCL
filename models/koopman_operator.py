import torch
import torch.nn as nn
from typing import Optional

class ConvEncoder1D(nn.Module):
    """把 [B, Nx] 编码到 latent 向量 [B, d]."""
    def __init__(self, nx: int, latent_dim: int, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
        )
        self.proj = nn.Linear(channels, latent_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = u.unsqueeze(1)           # [B, 1, Nx]
        h = self.net(x).squeeze(-1)  # [B, C]
        z = self.proj(h)             # [B, d]
        return z

class ConvDecoder1D(nn.Module):
    """把 latent 向量 [B, d] 解码回 [B, Nx]."""
    def __init__(self, nx: int, latent_dim: int, channels: int = 64):
        super().__init__()
        self.nx = nx
        self.fc = nn.Linear(latent_dim, channels * 16)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 先生成一个短序列，再插值到 Nx
        h = self.fc(z)                       # [B, C*16]
        B = h.shape[0]
        C = h.shape[1] // 16
        h = h.view(B, C, 16)                 # [B, C, 16]
        h = torch.nn.functional.interpolate(h, size=self.nx, mode="linear", align_corners=False)
        u = self.net(h).squeeze(1)           # [B, Nx]
        return u

class KoopmanRollout(nn.Module):
    """
    Koopman 潜空间线性演化：
      z_{t+1} = K z_t + B f
    其中 f 为可选的外加源项（控制），编码后进入潜空间。
    """
    def __init__(self, nx: int, latent_dim: int, enc_channels: int = 64, dec_channels: int = 64, use_forcing: bool = True):
        super().__init__()
        self.use_forcing = use_forcing
        self.encoder = ConvEncoder1D(nx, latent_dim, channels=enc_channels)
        self.decoder = ConvDecoder1D(nx, latent_dim, channels=dec_channels)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

        if use_forcing:
            self.force_encoder = ConvEncoder1D(nx, latent_dim, channels=enc_channels)
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
        u0: [B, Nx]
        returns:
          u_seq_pred: [B, steps, Nx] 预测序列 (从 t+1 开始)
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
