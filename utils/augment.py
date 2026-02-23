# utils/augment.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class AugmentConfig:
    num_ops_min: int = 1
    num_ops_max: int = 3

    # 1) additive noise
    noise_std_min: float = 0.0
    noise_std_max: float = 0.02

    # 2) spatial masking (drop segments)
    mask_ratio_min: float = 0.0
    mask_ratio_max: float = 0.15
    mask_block_min: int = 8
    mask_block_max: int = 32

    # 3) amplitude scaling & bias
    scale_min: float = 0.9
    scale_max: float = 1.1
    bias_min: float = -0.05
    bias_max: float = 0.05

    # 4) spectral filtering (random low-pass)
    #    用 FFT 随机截断高频，相当于“频域视角变化”
    k_cut_min: int = 8
    k_cut_max: int = 64

    # 控制输入 F(x) 是否也增强（通常不增强或只做很弱噪声）
    augment_control: bool = False
    control_noise_std: float = 0.005


def _rand_uniform(device, a: float, b: float) -> float:
    return (a + (b - a) * torch.rand((), device=device)).item()


def add_noise(u: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return u
    return u + std * torch.randn_like(u)


def amplitude_jitter(u: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    return u * scale + bias


def spatial_mask_1d(u: torch.Tensor, mask_ratio: float, block: int) -> torch.Tensor:
    """
    u: [T, X] or [B, T, X] or [X]
    用 block-wise mask 把若干空间片段置零（或置为均值也可以）
    """
    if mask_ratio <= 0:
        return u
    orig_shape = u.shape
    if u.dim() == 1:
        u_ = u[None, None, :]  # [1,1,X]
    elif u.dim() == 2:
        u_ = u[None, :, :]     # [1,T,X]
    else:
        u_ = u                 # [B,T,X]
    B, T, X = u_.shape
    num_mask = max(1, int(mask_ratio * X / max(1, block)))
    mask = torch.ones((B, 1, X), device=u.device, dtype=u.dtype)
    for b in range(B):
        for _ in range(num_mask):
            start = torch.randint(0, max(1, X - block + 1), (1,), device=u.device).item()
            mask[b, 0, start:start + block] = 0.0
    u_ = u_ * mask  # broadcast over T
    if len(orig_shape) == 1:
        return u_[0, 0, :]
    elif len(orig_shape) == 2:
        return u_[0, :, :]
    else:
        return u_


def spectral_lowpass(u: torch.Tensor, k_cut: int) -> torch.Tensor:
    """
    随机低通：保留 [0..k_cut] 的 rFFT 模态，其他置零后 irFFT。
    u: [T, X] or [B, T, X] or [X]
    """
    orig_shape = u.shape
    if u.dim() == 1:
        u_ = u[None, None, :]
    elif u.dim() == 2:
        u_ = u[None, :, :]
    else:
        u_ = u
    B, T, X = u_.shape
    # rfft over last dim
    U = torch.fft.rfft(u_, dim=-1)
    k_max = U.shape[-1] - 1
    k_cut = int(max(1, min(k_cut, k_max)))
    U[..., k_cut + 1:] = 0
    u_lp = torch.fft.irfft(U, n=X, dim=-1)
    if len(orig_shape) == 1:
        return u_lp[0, 0, :]
    elif len(orig_shape) == 2:
        return u_lp[0, :, :]
    else:
        return u_lp


def random_compose_view(
    u: torch.Tensor,
    f: Optional[torch.Tensor],
    cfg: AugmentConfig,
    rng: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
    """
    输入一条样本（u 序列，可带控制 f），输出增强后的 view + 记录
    u: [T, X] 或 [X]（按你工程里实际形状）
    f: [X] 或 None
    """
    device = u.device
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.randint(0, 2**31 - 1, (1,), device=device).item())

    ops = ["noise", "mask", "amp", "spec"]
    k_ops = int(torch.randint(cfg.num_ops_min, cfg.num_ops_max + 1, (1,), generator=rng, device=device).item())
    chosen = torch.randperm(len(ops), generator=rng, device=device)[:k_ops].tolist()

    info = {"ops": []}
    u_view = u
    f_view = f

    for idx in chosen:
        op = ops[idx]
        if op == "noise":
            std = _rand_uniform(device, cfg.noise_std_min, cfg.noise_std_max)
            u_view = add_noise(u_view, std)
            info["ops"].append(("noise", {"std": std}))
        elif op == "mask":
            ratio = _rand_uniform(device, cfg.mask_ratio_min, cfg.mask_ratio_max)
            block = int(_rand_uniform(device, cfg.mask_block_min, cfg.mask_block_max))
            u_view = spatial_mask_1d(u_view, ratio, block)
            info["ops"].append(("mask", {"ratio": ratio, "block": block}))
        elif op == "amp":
            scale = _rand_uniform(device, cfg.scale_min, cfg.scale_max)
            bias = _rand_uniform(device, cfg.bias_min, cfg.bias_max)
            u_view = amplitude_jitter(u_view, scale, bias)
            info["ops"].append(("amp", {"scale": scale, "bias": bias}))
        elif op == "spec":
            k_cut = int(_rand_uniform(device, cfg.k_cut_min, cfg.k_cut_max))
            u_view = spectral_lowpass(u_view, k_cut)
            info["ops"].append(("spec", {"k_cut": k_cut}))

    
    if cfg.augment_control and f_view is not None:
        f_view = f_view + cfg.control_noise_std * torch.randn_like(f_view)
        info["control"] = {"noise_std": cfg.control_noise_std}

    return u_view, f_view, info
