from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.diff import dt_forward
from utils.spectral_diff import spectral_ddx, spectral_ddxx

@dataclass
class LibraryConfig:
    include_u: bool = True
    include_ux: bool = True
    include_uxx: bool = True
    include_u2: bool = True
    include_uux: bool = True  # u * u_x
    include_forcing: bool = True  # 强迫项 F(x)

class PhysicsLibrary(nn.Module):
    """
    构建候选库 Θ(u)，用于显式 PDE 回归：
      u_t ≈ Θ(u) ξ
    """
    def __init__(self, cfg: LibraryConfig):
        super().__init__()
        self.cfg = cfg

    def term_names(self) -> List[str]:
        names = []
        if self.cfg.include_u: names.append("u")
        if self.cfg.include_ux: names.append("u_x")
        if self.cfg.include_uxx: names.append("u_xx")
        if self.cfg.include_u2: names.append("u^2")
        if self.cfg.include_uux: names.append("u * u_x")
        if self.cfg.include_forcing: names.append("F")  # 强迫项
        return names

    def forward(self, u: torch.Tensor, dx: float, dt: Optional[float] = None, u_seq: Optional[torch.Tensor] = None, F: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        u: [B, T, Nx] or [B, Nx]
        dx: float - 空间步长
        dt: float - 时间步长（用于Phase B的中心差分）
        u_seq: [B, T, Nx] - 完整序列（用于Phase B的中心差分）
        F: [B, Nx] or [B, 1, Nx] - 强迫项
        returns:
          Theta: [..., K]  (最后一维为候选项数)
          cache:  保存导数等中间量，便于调试/可解释输出
        """
        orig_shape = u.shape
        if u.dim() == 2:
            u2 = u.unsqueeze(1)  # [B, 1, Nx]
        else:
            u2 = u

        # 使用谱导数计算空间导数
        ux = spectral_ddx(u2, dx)
        uxx = spectral_ddxx(u2, dx)

        terms = []
        cache = {"u": u2, "u_x": ux, "u_xx": uxx}

        if self.cfg.include_u:
            terms.append(u2)
        if self.cfg.include_ux:
            terms.append(ux)
        if self.cfg.include_uxx:
            terms.append(uxx)
        if self.cfg.include_u2:
            terms.append(u2 ** 2)
        if self.cfg.include_uux:
            terms.append(u2 * ux)
        if self.cfg.include_forcing and F is not None:
            # 确保F的形状与u2匹配：[B, Nx] -> [B, 1, Nx]
            F_expanded = F.unsqueeze(1) if F.dim() == 2 else F
            F_expanded = F_expanded.expand_as(u2)  # [B, T, Nx]
            terms.append(F_expanded)
            cache["F"] = F_expanded

        Theta = torch.stack(terms, dim=-1)  # [B, T, Nx, K]
        return Theta, cache

class PhysicsRegressor(nn.Module):
    """
    学习稀疏系数 ξ，使得:
      u_t ≈ Θ(u) ξ
    ξ 是全局共享的可学习参数，代表系统本身的物理结构。
    通过投影器约束ξ的可行集合到物理上合理的区域。
    """
    def __init__(self, library: PhysicsLibrary, projector_type: str = "none", 
                 threshold: float = 1e-3, top_k: int = None, gate_init: float = 0.5):
        super().__init__()
        self.library = library
        K = len(self.library.term_names())
        self.K = K
        
        # ξ 是全局共享的可学习参数，不依赖具体样本
        # 对所有 batch、所有时间、所有空间点是同一组参数
        # 初始化为小的随机值以打破对称性
        self.xi = nn.Parameter(torch.randn(K) * 0.1)
        
        # 投影器配置
        self.projector_type = projector_type
        self.threshold = threshold
        self.top_k = top_k
        
        # Gate/mask 投影器参数
        if projector_type == "gate":
            self.gate_logits = nn.Parameter(torch.full((K,), gate_init))
        elif projector_type == "hard-concrete":
            self.gate_logits = nn.Parameter(torch.full((K,), gate_init))
            self.temp = 0.1  # 温度参数，用于Gumbel-Softmax

    def _project_xi(self):
        """
        对ξ应用投影器，约束其到物理可行集合Hphys
        """
        xi_proj = self.xi.clone()
        
        if self.projector_type == "hard-threshold":
            # Hard threshold: 低于阈值的系数置为0
            xi_proj = torch.where(torch.abs(xi_proj) < self.threshold, torch.zeros_like(xi_proj), xi_proj)
        
        elif self.projector_type == "top-k":
            # Top-k: 只保留绝对值最大的k个系数
            if self.top_k is not None and self.top_k < self.K:
                _, indices = torch.topk(torch.abs(xi_proj), self.top_k, largest=True)
                mask = torch.zeros_like(xi_proj)
                mask[indices] = 1.0
                xi_proj = xi_proj * mask
        
        elif self.projector_type == "gate":
            # Gate/mask: 使用可学习的门控参数
            gate = torch.sigmoid(self.gate_logits)
            xi_proj = xi_proj * gate
        
        elif self.projector_type == "hard-concrete":
            # Hard-concrete gate (L0 regularization)
            # 参考: https://arxiv.org/abs/1712.01312
            u = torch.rand_like(self.gate_logits)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.gate_logits) / self.temp)
            s = s * (1.0 - 1e-3) + 1e-3 / 2.0  # 防止s为0或1
            gate = torch.sigmoid((torch.log(s) - torch.log(1 - s)) / self.temp)
            xi_proj = xi_proj * gate
        
        return xi_proj

    def forward(self, u: torch.Tensor, dx: float, u_t: Optional[torch.Tensor] = None, F: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u: [B, T, Nx]
        u_t: [B, T, Nx] (可选) 真实的时间导数，用于标准化
        F: [B, Nx] or [B, 1, Nx] (可选) 强迫项
        returns:
          u_t_hat: [B, T, Nx]  由 Θ(u) ξ 给出的时间导数估计
          Theta:   [B, T, Nx, K]
        """
        Theta, _ = self.library(u, dx, F=F)
        B, T, Nx, K = Theta.shape
        
        # 对 Θ 的每一列做标准化
        epsilon = 1e-8  # 防止除以零
        
        # 计算每列的L2范数：[B, T, K]
        Theta_norms = torch.norm(Theta, p=2, dim=2, keepdim=False)  # [B, T, K]
        Theta_norms = Theta_norms + epsilon  # [B, T, K]
        
        # 标准化 Θ：[B, T, Nx, K] / [B, T, 1, K] -> [B, T, Nx, K]
        Theta_normalized = Theta / Theta_norms.unsqueeze(2).expand_as(Theta)
        
        # 应用投影器约束ξ到物理可行集合
        xi_proj = self._project_xi()
        
        # 使用投影后的 ξ 参数计算时间导数
        # Theta: [B, T, Nx, K], xi_proj: [K] -> u_t_hat: [B, T, Nx]
        u_t_hat_normalized = torch.einsum("btnk,k->btn", Theta_normalized, xi_proj)
        
        # 如果提供了 u_t，将 u_t_hat 还原到原始尺度
        if u_t is not None:
            # 计算 u_t 的均值和标准差：[B, T, 1]
            u_t_mean = u_t.mean(dim=2, keepdim=True)
            u_t_std = u_t.std(dim=2, keepdim=True) + epsilon
            u_t_hat = u_t_hat_normalized * u_t_std + u_t_mean
        else:
            # 否则，使用 Θ 的范数还原尺度
            u_t_hat = torch.einsum("btnk,k,btnk->btn", Theta, xi_proj, Theta_normalized)
        
        return u_t_hat, Theta

    def pretty_equation(self, thresh: float = 1e-3) -> str:
        """
        使用当前的全局参数 ξ 生成可读的物理方程。
        
        Args:
            thresh: 系数阈值，低于此值的项将被忽略
            
        Returns:
            可读的物理方程字符串
        """
        # 使用投影后的 ξ 参数
        with torch.no_grad():
            xi_proj = self._project_xi()
            xi_list = xi_proj.detach().cpu().tolist()
        
        names = self.library.term_names()
        parts = []
        for c, n in zip(xi_list, names):
            if abs(c) >= thresh:
                parts.append(f"{c:+.4f} * {n}")
        
        if not parts:
            return "u_t = 0 (all coefficients below threshold)"
        
        rhs = " ".join(parts).replace("+ -", "- ")
        return f"u_t = {rhs}"
        
    def get_l1_regularization(self, lambda_l1: float) -> torch.Tensor:
        """
        计算L1正则化损失，用于诱导系数的稀疏性。
        
        Args:
            lambda_l1: L1正则化权重
            
        Returns:
            L1正则化损失
        """
        # 直接对全局 ξ 参数计算L1损失
        return lambda_l1 * self.xi.abs().sum()
