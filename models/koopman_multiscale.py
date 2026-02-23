import torch
import torch.nn as nn
from typing import Optional

from models.koopman_operator import ConvEncoder1D, ConvDecoder1D


def _rfft_band(u: torch.Tensor, k_min: int, k_max: int) -> torch.Tensor:
    """在频域截取 [k_min, k_max] 频带并反变换回物理域。

    说明：这里使用 rFFT（实信号），频率索引 k 对应离散模态。
    u: [B, Nx]
    return: [B, Nx]
    """
    B, Nx = u.shape
    U = torch.fft.rfft(u, dim=-1)  # [B, Nx//2+1]
    # 频率索引范围
    k_max = min(k_max, U.shape[-1] - 1)
    k_min = max(k_min, 0)
    mask = torch.zeros_like(U, dtype=torch.bool)
    mask[..., k_min : k_max + 1] = True
    U_band = torch.where(mask, U, torch.zeros_like(U))
    u_band = torch.fft.irfft(U_band, n=Nx, dim=-1)
    return u_band


def split_multiscale(u: torch.Tensor, k_low: int, k_mid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把信号拆成 (low, mid, high) 三个尺度。

    - low: 频率 0..k_low
    - mid: (k_low+1)..k_mid
    - high: (k_mid+1)..Nyquist

    u: [B, Nx]
    return: (u_low, u_mid, u_high), 形状均为 [B, Nx]
    """
    B, Nx = u.shape
    nfreq = Nx // 2  # rfft 最大频率索引约为 Nx//2
    k_low = int(k_low)
    k_mid = int(k_mid)
    k_low = max(0, min(k_low, nfreq))
    k_mid = max(k_low, min(k_mid, nfreq))

    u_low = _rfft_band(u, 0, k_low)
    u_mid = _rfft_band(u, k_low + 1, k_mid) if k_mid >= k_low + 1 else torch.zeros_like(u)
    u_high = u - u_low - u_mid
    return u_low, u_mid, u_high


class HighFreqResidualNet(nn.Module):
    """高频补偿网络：预测 high_{t+1} 或 delta_high。

    设计目标：
    - 低/中频由 Koopman 线性演化负责；
    - 高频由小型卷积残差网络补偿，以提升激波/尖峰等细节的滚动稳定性。
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, u_high: torch.Tensor, u_ctx: torch.Tensor) -> torch.Tensor:
        """u_high: [B, Nx] 高频分量
        u_ctx:  [B, Nx] 作为上下文的全信号或低中频预测
        return: [B, Nx] 预测的 high_{t+1}
        """
        x = torch.stack([u_high, u_ctx], dim=1)  # [B, 2, Nx]
        y = self.net(x).squeeze(1)               # [B, Nx]
        return y


class MultiScaleKoopmanRollout(nn.Module):
    """多尺度 Koopman 演化算子（低/中频线性演化 + 高频残差补偿）。

    低频与中频分别拥有各自的 (encoder, K, decoder)，对应不同尺度的可线性化潜空间。
    高频使用残差网络预测下一步高频分量，从而提升尖峰/激波细节的可滚动预测稳定性。

    接口与 KoopmanRollout 保持一致：step / rollout / encode。
    """
    def __init__(
        self,
        nx: int,
        latent_dim_low: int,
        latent_dim_mid: int,
        k_low: int,
        k_mid: int,
        enc_channels: int = 64,
        dec_channels: int = 64,
        high_channels: int = 64,
        use_forcing: bool = True,
    ):
        super().__init__()
        self.nx = int(nx)
        self.k_low = int(k_low)
        self.k_mid = int(k_mid)
        self.use_forcing = bool(use_forcing)

        # Low-scale Koopman
        self.enc_low = ConvEncoder1D(nx, latent_dim_low, channels=enc_channels)
        self.dec_low = ConvDecoder1D(nx, latent_dim_low, channels=dec_channels)
        self.K_low = nn.Linear(latent_dim_low, latent_dim_low, bias=False)

        # Mid-scale Koopman
        self.enc_mid = ConvEncoder1D(nx, latent_dim_mid, channels=enc_channels)
        self.dec_mid = ConvDecoder1D(nx, latent_dim_mid, channels=dec_channels)
        self.K_mid = nn.Linear(latent_dim_mid, latent_dim_mid, bias=False)

        # Optional forcing on low/mid
        if self.use_forcing:
            self.force_enc_low = ConvEncoder1D(nx, latent_dim_low, channels=enc_channels)
            self.force_enc_mid = ConvEncoder1D(nx, latent_dim_mid, channels=enc_channels)
            self.B_low = nn.Linear(latent_dim_low, latent_dim_low, bias=False)
            self.B_mid = nn.Linear(latent_dim_mid, latent_dim_mid, bias=False)
        else:
            self.force_enc_low = None
            self.force_enc_mid = None
            self.B_low = None
            self.B_mid = None

        # High-frequency compensator
        self.high_net = HighFreqResidualNet(channels=high_channels)

    def _split(self, u: torch.Tensor):
        return split_multiscale(u, k_low=self.k_low, k_mid=self.k_mid)

    def step(self, u_t: torch.Tensor, F: Optional[torch.Tensor] = None):
        """单步预测。

        u_t: [B, Nx]
        F:   [B, Nx] 外加源项（可选）
        return:
          u_next: [B, Nx]
          z_cat:  [B, d_low + d_mid] 仅用于一致性等用途
        """
        u_low, u_mid, u_high = self._split(u_t)

        z_low = self.enc_low(u_low)
        z_mid = self.enc_mid(u_mid)

        if self.use_forcing and (F is not None):
            f_low, f_mid, _ = self._split(F)
            fz_low = self.force_enc_low(f_low)
            fz_mid = self.force_enc_mid(f_mid)
            z_low_next = self.K_low(z_low) + self.B_low(fz_low)
            z_mid_next = self.K_mid(z_mid) + self.B_mid(fz_mid)
        else:
            z_low_next = self.K_low(z_low)
            z_mid_next = self.K_mid(z_mid)

        u_low_next = self.dec_low(z_low_next)
        u_mid_next = self.dec_mid(z_mid_next)

        # 高频：用 (u_high, u_low_next+u_mid_next) 作为上下文，预测 high_{t+1}
        u_ctx = u_low_next + u_mid_next
        u_high_next = self.high_net(u_high, u_ctx)
        u_next = u_ctx + u_high_next

        z_cat = torch.cat([z_low_next, z_mid_next], dim=-1)
        return u_next, z_cat

    def rollout(self, u0: torch.Tensor, steps: int, F: Optional[torch.Tensor] = None):
        preds = []
        u = u0
        for _ in range(int(steps)):
            u, _ = self.step(u, F)
            preds.append(u)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def encode(self, u: torch.Tensor) -> torch.Tensor:
        # 用 low/mid 的潜空间拼接表示整体（用于一致性正则）
        u_low, u_mid, _ = self._split(u)
        z_low = self.enc_low(u_low)
        z_mid = self.enc_mid(u_mid)
        return torch.cat([z_low, z_mid], dim=-1)
