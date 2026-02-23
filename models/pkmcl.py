from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.koopman_operator import KoopmanRollout
from models.koopman_multiscale import MultiScaleKoopmanRollout
from models.physics_library import PhysicsLibrary, PhysicsRegressor, LibraryConfig
from utils.diff import dt_forward

@dataclass
class PKMCLConfig:
    nx: int
    latent_dim: int
    enc_channels: int
    dec_channels: int
    use_forcing: bool
    lib_cfg: LibraryConfig

    # Multi-scale Koopman (可选)
    multiscale: bool = False
    k_low: int = 12
    k_mid: int = 48
    latent_dim_low: int = 64
    latent_dim_mid: int = 64
    high_channels: int = 64
    
    # Physics Regressor (可选)
    projector_type: str = "gate"
    gate_init: float = 0.5
    threshold: float = 1e-3
    
    # Physics Regressor (可选)
    projector_type: str = "none"
    threshold: float = 1e-3
    top_k: Optional[int] = None
    gate_init: float = 0.5

class PKMCL(nn.Module):
    """
    输出两类对象：
      (1) KoopmanRollout：可滚动预测算子
      (2) PhysicsRegressor：显式 PDE 稀疏方程 (Θ(u) ξ)
    可选一致性（BYOL-style）：用 EMA target 网络约束潜空间表示稳定。
    """
    def __init__(self, cfg: PKMCLConfig):
        super().__init__()
        self.cfg = cfg
        if bool(cfg.multiscale):
            self.dyn = MultiScaleKoopmanRollout(
                nx=cfg.nx,
                latent_dim_low=int(cfg.latent_dim_low),
                latent_dim_mid=int(cfg.latent_dim_mid),
                k_low=int(cfg.k_low),
                k_mid=int(cfg.k_mid),
                enc_channels=cfg.enc_channels,
                dec_channels=cfg.dec_channels,
                high_channels=int(cfg.high_channels),
                use_forcing=cfg.use_forcing,
            )
        else:
            self.dyn = KoopmanRollout(
                nx=cfg.nx, latent_dim=cfg.latent_dim,
                enc_channels=cfg.enc_channels, dec_channels=cfg.dec_channels,
                use_forcing=cfg.use_forcing
            )
        self.library = PhysicsLibrary(cfg.lib_cfg)
        # 传递物理回归器参数
        phys_kwargs = {}
        if hasattr(cfg, 'projector_type'):
            phys_kwargs['projector_type'] = cfg.projector_type
        if hasattr(cfg, 'gate_init'):
            phys_kwargs['gate_init'] = cfg.gate_init
        if hasattr(cfg, 'threshold'):
            phys_kwargs['threshold'] = cfg.threshold
        self.phys = PhysicsRegressor(self.library, **phys_kwargs)

        # target 网络（仅用于一致性损失，可选）
        self.dyn_tgt = None

    def build_target(self):
        # 建立 target 网络（参数复制）
        if bool(self.cfg.multiscale):
            self.dyn_tgt = MultiScaleKoopmanRollout(
                nx=self.cfg.nx,
                latent_dim_low=int(self.cfg.latent_dim_low),
                latent_dim_mid=int(self.cfg.latent_dim_mid),
                k_low=int(self.cfg.k_low),
                k_mid=int(self.cfg.k_mid),
                enc_channels=self.cfg.enc_channels,
                dec_channels=self.cfg.dec_channels,
                high_channels=int(self.cfg.high_channels),
                use_forcing=self.cfg.use_forcing,
            )
        else:
            self.dyn_tgt = KoopmanRollout(
                nx=self.cfg.nx, latent_dim=self.cfg.latent_dim,
                enc_channels=self.cfg.enc_channels, dec_channels=self.cfg.dec_channels,
                use_forcing=self.cfg.use_forcing
            )
        self.dyn_tgt.load_state_dict(self.dyn.state_dict())
        for p in self.dyn_tgt.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, tau: float):
        if self.dyn_tgt is None:
            return
        for p_tgt, p in zip(self.dyn_tgt.parameters(), self.dyn.parameters()):
            p_tgt.data.mul_(tau).add_((1 - tau) * p.data)

    def rollout(self, u0: torch.Tensor, steps: int, Fsrc: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.dyn.rollout(u0, steps=steps, F=Fsrc)

    def loss_rec(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(u_pred, u_true)

    def loss_phy(self, u_pred_seq: torch.Tensor, dt: float, dx: float, F: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        物理损失：|| u_t - Θ(u) ξ ||^2
        使用预测序列 u_pred_seq 来计算 u_t（时间差分）与 Θ(u_pred_seq[:-1]) ξ
        """
        # u_pred_seq: [B, T, Nx] 这里 T=rollout_steps
        # 用 forward difference 得到 [B, T-1, Nx]
        u_t = dt_forward(u_pred_seq, dt=dt)
        # Θ(u) ξ 以 u_pred_seq[:-1] 计算
        u_for_lib = u_pred_seq[:, :-1, :]
        # 将真实的u_t传递给phys.forward()以进行正确的标准化
        u_t_hat, _ = self.phys(u_for_lib, dx=dx, u_t=u_t, F=F)
        return F.mse_loss(u_t_hat, u_t), u_t_hat

    def loss_con(self, u0: torch.Tensor, Fsrc: Optional[torch.Tensor] = None, u_view1: Optional[torch.Tensor] = None, u_view2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        BYOL-style一致性损失：
        1. 如果提供了两个预增强的视图（u_view1, u_view2），则使用它们计算一致性损失
        2. 否则，对原始输入u0进行简单的高斯噪声增强
        """
        assert self.dyn_tgt is not None, "Call build_target() before using consistency loss."
        
        if u_view1 is not None and u_view2 is not None:
            # 使用预增强的视图进行BYOL-style训练
            # 视图1通过online网络，视图2通过target网络
            z_online = self.dyn.encode(u_view1)
            with torch.no_grad():
                z_tgt = self.dyn_tgt.encode(u_view2)
        else:
            # 回退到原始的随机噪声增强方式
            noise = 0.01 * torch.randn_like(u0)
            u_aug = u0 + noise

            z_online = self.dyn.encode(u0)
            with torch.no_grad():
                z_tgt = self.dyn_tgt.encode(u_aug)
        
        z_online = F.normalize(z_online, dim=-1)
        z_tgt = F.normalize(z_tgt, dim=-1)
        return F.mse_loss(z_online, z_tgt)

    def discovered_equation(self, u: torch.Tensor, dx: float, thresh: float = 1e-3) -> str:
        """
        输出发现的物理方程。
        
        Args:
            u: 样例输入 [B, T, Nx] 或 [B, Nx] (不再使用)
            dx: 空间步长 (不再使用)
            thresh: 系数阈值，低于此值的项将被忽略
            
        Returns:
            可读的物理方程字符串
        """
        return self.phys.pretty_equation(thresh=thresh)
