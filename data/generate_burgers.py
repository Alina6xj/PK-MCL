import argparse
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm

# 复制utils/augment.py中的必要代码
@dataclass
class AugmentConfig:
    # 总体：每次对一个样本随机挑若干增强串联（k 次）
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

    # 控制输入一般不增强，除非你论文明确
    if cfg.augment_control and f_view is not None:
        f_view = f_view + cfg.control_noise_std * torch.randn_like(f_view)
        info["control"] = {"noise_std": cfg.control_noise_std}

    return u_view, f_view, info

def _rand_ic_fourier(x: np.ndarray, n_modes: int, amp: float) -> np.ndarray:
    """随机 Fourier 叠加初值（周期边界）。"""
    u0 = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        a = np.random.randn()
        b = np.random.randn()
        u0 += a * np.sin(2 * np.pi * k * x) + b * np.cos(2 * np.pi * k * x)
    u0 = amp * u0 / (np.max(np.abs(u0)) + 1e-12)
    return u0

def _rand_forcing(x: np.ndarray, amp: float, n_gauss: int) -> np.ndarray:
    """随机源项 F(x)：若干个 Gaussian 的叠加（周期区间上）。"""
    F = np.zeros_like(x)
    for _ in range(n_gauss):
        mu = np.random.rand()
        sigma = 0.03 + 0.07 * np.random.rand()
        sign = 1.0 if np.random.rand() > 0.5 else -1.0
        F += sign * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    F = amp * F / (np.max(np.abs(F)) + 1e-12)
    return F

def _ddx(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)

def _ddxx(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

def _rhs_burgers(u: np.ndarray, dx: float, nu: float, F: np.ndarray) -> np.ndarray:
    """u_t = -u u_x + nu u_xx + F(x)"""
    ux = _ddx(u, dx)
    uxx = _ddxx(u, dx)
    return (-u * ux) + nu * uxx + F

def simulate_burgers(
    nx: int,
    dt: float,
    t_end: float,
    snapshot_dt: float,
    nu: float,
    forcing: bool,
    forcing_amp: float,
    forcing_n_gauss: int,
    ic_n_modes: int,
    ic_amp: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用周期边界 + 中心差分 + RK2(Heun) 生成一条轨迹。
    返回:
      traj: [T, Nx]
      F:    [Nx]
    """
    L = 1.0
    x = np.linspace(0.0, L, nx, endpoint=False)
    dx = x[1] - x[0]

    u = _rand_ic_fourier(x, ic_n_modes, ic_amp)
    F = _rand_forcing(x, forcing_amp, forcing_n_gauss) if forcing else np.zeros_like(x)

    n_steps = int(np.round(t_end / dt))
    snap_every = int(np.round(snapshot_dt / dt))
    assert snap_every >= 1

    snaps = []
    t = 0.0
    for n in range(n_steps + 1):
        if n % snap_every == 0:
            snaps.append(u.copy())
        # RK2
        k1 = _rhs_burgers(u, dx, nu, F)
        u1 = u + dt * k1
        k2 = _rhs_burgers(u1, dx, nu, F)
        u = 0.5 * u + 0.5 * (u1 + dt * k2)
        t += dt

    traj = np.stack(snaps, axis=0)
    meta = {"L": L, "dx": dx}
    return traj, F, meta

def build_dataset(
    n_traj: int,
    n_train: int,
    nx: int,
    dt: float,
    t_end: float,
    snapshot_dt: float,
    nu: float,
    forcing: bool,
    forcing_amp: float,
    forcing_n_gauss: int,
    ic_n_modes: int,
    ic_amp: float,
    augment_cfg: Optional[AugmentConfig] = None,
) -> Dict[str, Any]:
    trajs = []
    forces = []
    view1_list = []
    view2_list = []
    meta0 = None
    
    for _ in tqdm(range(n_traj), desc="Simulating Burgers trajectories"):
        traj, F, meta = simulate_burgers(
            nx=nx, dt=dt, t_end=t_end, snapshot_dt=snapshot_dt, nu=nu,
            forcing=forcing, forcing_amp=forcing_amp, forcing_n_gauss=forcing_n_gauss,
            ic_n_modes=ic_n_modes, ic_amp=ic_amp
        )
        
        # 转换为torch张量
        traj_tensor = torch.tensor(traj, dtype=torch.float32)  # [T, Nx]
        F_tensor = torch.tensor(F, dtype=torch.float32)        # [Nx]
        
        trajs.append(traj)
        forces.append(F)
        meta0 = meta0 or meta
        
        # 如果配置了数据增强，生成两个视图
        if augment_cfg is not None:
            # 生成视图1
            view1_u, view1_f, _ = random_compose_view(traj_tensor, F_tensor, augment_cfg)
            # 生成视图2
            view2_u, view2_f, _ = random_compose_view(traj_tensor, F_tensor, augment_cfg)
            
            view1_list.append(view1_u)
            view2_list.append(view2_u)

    trajs = torch.tensor(np.stack(trajs, axis=0), dtype=torch.float32)   # [N, T, Nx]
    forces = torch.tensor(np.stack(forces, axis=0), dtype=torch.float32) # [N, Nx]
    
    # 处理视图数据
    view1 = torch.stack(view1_list, dim=0) if view1_list else None  # [N, T, Nx]
    view2 = torch.stack(view2_list, dim=0) if view2_list else None  # [N, T, Nx]

    train = {"u": trajs[:n_train], "F": forces[:n_train]}
    test  = {"u": trajs[n_train:], "F": forces[n_train:]}
    
    # 划分视图数据
    train_view1 = view1[:n_train] if view1 is not None else None
    train_view2 = view2[:n_train] if view2 is not None else None
    test_view1 = view1[n_train:] if view1 is not None else None
    test_view2 = view2[n_train:] if view2 is not None else None

    meta = {
        "nx": nx, "dt": dt, "t_end": t_end, "snapshot_dt": snapshot_dt,
        "nu": nu, "forcing": forcing, "dx": meta0["dx"], "L": meta0["L"],
        "n_traj": n_traj, "n_train": n_train,
        "augmented": augment_cfg is not None
    }

    result = {
        "train_u": train["u"], "train_F": train["F"],
        "test_u": test["u"], "test_F": test["F"],
        "meta": meta
    }
    
    # 添加视图数据
    if augment_cfg is not None:
        result.update({
            "train_view1": train_view1,
            "train_view2": train_view2,
            "test_view1": test_view1,
            "test_view2": test_view2
        })
    
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/burgers_dataset.pt")
    p.add_argument("--n_traj", type=int, default=120)
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--t_end", type=float, default=1.0)
    p.add_argument("--snapshot_dt", type=float, default=1e-2)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--forcing", action="store_true")
    p.add_argument("--forcing_amp", type=float, default=0.5)
    p.add_argument("--forcing_n_gauss", type=int, default=2)
    p.add_argument("--ic_n_modes", type=int, default=8)
    p.add_argument("--ic_amp", type=float, default=1.0)
    
    # 数据增强参数
    p.add_argument("--augment", action="store_true", help="启用数据增强生成视图")
    p.add_argument("--num_ops_min", type=int, default=1, help="每次增强的最小操作数")
    p.add_argument("--num_ops_max", type=int, default=3, help="每次增强的最大操作数")
    p.add_argument("--noise_std_max", type=float, default=0.02, help="噪声标准差最大值")
    p.add_argument("--mask_ratio_max", type=float, default=0.15, help="掩码比例最大值")
    
    args = p.parse_args()
    
    # 创建数据增强配置
    augment_cfg = None
    if args.augment:
        augment_cfg = AugmentConfig(
            num_ops_min=args.num_ops_min,
            num_ops_max=args.num_ops_max,
            noise_std_max=args.noise_std_max,
            mask_ratio_max=args.mask_ratio_max
        )

    ds = build_dataset(
        n_traj=args.n_traj, n_train=args.n_train,
        nx=args.nx, dt=args.dt, t_end=args.t_end, snapshot_dt=args.snapshot_dt,
        nu=args.nu, forcing=args.forcing, forcing_amp=args.forcing_amp,
        forcing_n_gauss=args.forcing_n_gauss, ic_n_modes=args.ic_n_modes, ic_amp=args.ic_amp,
        augment_cfg=augment_cfg
    )
    torch.save(ds, args.out)
    print(f"Saved dataset to {args.out}")
    print("Meta:", ds["meta"])
    
    # 打印视图信息
    if "train_view1" in ds:
        print(f"Generated views: train_view1 shape={ds['train_view1'].shape}, train_view2 shape={ds['train_view2'].shape}")
        print(f"                 test_view1 shape={ds['test_view1'].shape}, test_view2 shape={ds['test_view2'].shape}")

if __name__ == "__main__":
    main()
