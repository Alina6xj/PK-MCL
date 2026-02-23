from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.koopman_operator import KoopmanRollout
from models.koopman_multiscale import MultiScaleKoopmanRollout
from models.physics_library import LibraryConfig
from models.physics_library_autodiff import PhysicsLibraryAutodiff, PhysicsRegressorAutodiff
from utils.diff import dt_forward

@dataclass
class PKMCLAutoDiffConfig:
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
    top_k: Optional[int] = None
    
    # 自动微分配置
    mlp_hidden_dim: int = 64
    mlp_num_layers: int = 3

class PKMCLAutoDiff(nn.Module):
    """
    使用自动微分的PKMCL模型
    输出两类对象：
      (1) KoopmanRollout：可滚动预测算子
      (2) PhysicsRegressor：显式 PDE 稀疏方程 (Θ(u) ξ)
    使用自动微分替代有限差分计算导数
    """
    def __init__(self, cfg: PKMCLAutoDiffConfig):
        super().__init__()
        self.cfg = cfg
        
        # 初始化Koopman滚动预测器
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
        
        # 使用自动微分物理库
        lib_cfg_dict = {
            "include_u": cfg.lib_cfg.include_u,
            "include_ux": cfg.lib_cfg.include_ux,
            "include_uxx": cfg.lib_cfg.include_uxx,
            "include_u2": cfg.lib_cfg.include_u2,
            "include_uux": cfg.lib_cfg.include_uux
        }
        
        self.library = PhysicsLibraryAutodiff(
            lib_cfg_dict,
            mlp_hidden_dim=cfg.mlp_hidden_dim,
            mlp_num_layers=cfg.mlp_num_layers
        )
        
        # 传递物理回归器参数
        phys_kwargs = {
            "projector_type": cfg.projector_type,
            "gate_init": cfg.gate_init,
            "threshold": cfg.threshold,
            "top_k": cfg.top_k
        }
        
        self.phys = PhysicsRegressorAutodiff(self.library, **phys_kwargs)

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

    def loss_phy(self, u_pred_seq: torch.Tensor, dt: float, dx: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        物理损失：|| u_t - Θ(u) ξ ||^2
        使用预测序列 u_pred_seq 来计算 u_t（时间差分）与 Θ(u_pred_seq[:-1]) ξ
        使用自动微分计算导数
        """
        # u_pred_seq: [B, T, Nx] 这里 T=rollout_steps
        # 用 forward difference 得到 [B, T-1, Nx]
        u_t = dt_forward(u_pred_seq, dt=dt)
        # Θ(u) ξ 以 u_pred_seq[:-1] 计算
        u_for_lib = u_pred_seq[:, :-1, :]
        
        # 创建归一化到[-1, 1]的坐标网格
        B, T_lib, Nx = u_for_lib.shape
        
        # 归一化x坐标到[-1, 1]
        x = torch.linspace(-1, 1, Nx, device=u_for_lib.device)
        x = x.view(1, 1, Nx, 1).expand(B, T_lib, Nx, 1)
        
        # 归一化t坐标到[-1, 1]
        t = torch.linspace(-1, 1, T_lib, device=u_for_lib.device)
        t = t.view(1, T_lib, 1, 1).expand(B, T_lib, Nx, 1)
        
        # 拼接坐标 [B, T_lib, Nx, 2]
        coords = torch.cat([x, t], dim=-1)
        
        # 使用自动微分物理回归器计算u_t_hat
        u_t_hat, _ = self.phys(u_for_lib, coords, u_t=u_t)
        
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
        """
        return self.phys.pretty_equation(thresh=thresh)
