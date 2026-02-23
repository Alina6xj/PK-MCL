import argparse
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm


def _wavepacket_initial(x: np.ndarray, A: float, x0: float, sigma: float, k0: float) -> np.ndarray:
    """
    生成波包初始条件
    ψ(x,0) = A exp[-(x-x0)²/(2σ²)] exp(i k0 x)
    """
    gaussian = A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    phase = np.exp(1j * k0 * x)
    return gaussian * phase


def _random_wavepacket(x: np.ndarray) -> np.ndarray:
    """
    生成随机波包初始条件
    A ∈ [0.8, 1.2], x0 ~ U[-10, 10], σ=2.0, k0 ~ U[0, 2]
    """
    A = np.random.uniform(0.8, 1.2)
    x0 = np.random.uniform(-10, 10)
    sigma = 2.0
    k0 = np.random.uniform(0, 2)
    return _wavepacket_initial(x, A, x0, sigma, k0)


def _split_step_fourier_step(psi: np.ndarray, k: np.ndarray, dt: float, kappa: float) -> np.ndarray:
    """
    单步分裂步傅里叶方法
    """
    psi_hat = np.fft.fft(psi)
    
    psi_hat_half = psi_hat * np.exp(-0.5 * 1j * k**2 * dt)
    
    psi_half = np.fft.ifft(psi_hat_half)
    
    psi_half = psi_half * np.exp(-1j * kappa * np.abs(psi_half)**2 * dt)
    
    psi_hat_full = np.fft.fft(psi_half)
    
    psi_hat_full = psi_hat_full * np.exp(-0.5 * 1j * k**2 * dt)
    
    psi_full = np.fft.ifft(psi_hat_full)
    
    return psi_full


def simulate_schrodinger(
    nx: int = 512,
    x_min: float = -20.0,
    x_max: float = 20.0,
    dt: float = 0.005,
    t_end: float = 5.0,
    snapshot_dt: float = 0.05,
    kappa: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用分裂步傅里叶方法模拟1D非线性薛定谔方程
    i ∂_t ψ = -∇² ψ - κ |ψ|² ψ
    返回:
      traj_psi_real: [T, Nx] - 实部
      traj_psi_imag: [T, Nx] - 虚部
      meta: 元数据
    """
    L = x_max - x_min
    x = np.linspace(x_min, x_max, nx, endpoint=False)
    dx = x[1] - x[0]
    
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    
    psi = _random_wavepacket(x)
    
    n_steps = int(np.round(t_end / dt))
    snap_every = int(np.round(snapshot_dt / dt))
    assert snap_every >= 1
    
    snaps_real = []
    snaps_imag = []
    t = 0.0
    
    for n in range(n_steps + 1):
        if n % snap_every == 0:
            snaps_real.append(np.real(psi).copy())
            snaps_imag.append(np.imag(psi).copy())
        
        psi = _split_step_fourier_step(psi, k, dt, kappa)
        
        t += dt
    
    traj_real = np.stack(snaps_real, axis=0)
    traj_imag = np.stack(snaps_imag, axis=0)
    
    meta = {
        "L": L,
        "dx": dx,
        "x_min": x_min,
        "x_max": x_max,
        "nx": nx,
        "kappa": kappa
    }
    
    return traj_real, traj_imag, meta


def build_dataset(
    n_traj: int = 100,
    n_train: int = 80,
    nx: int = 512,
    x_min: float = -20.0,
    x_max: float = 20.0,
    dt: float = 0.005,
    t_end: float = 5.0,
    snapshot_dt: float = 0.05,
    kappa: float = 1.0,
) -> Dict[str, Any]:
    """
    构建薛定谔方程数据集
    """
    trajs_real = []
    trajs_imag = []
    meta0 = None
    
    for _ in tqdm(range(n_traj), desc="Simulating Schrodinger trajectories"):
        traj_real, traj_imag, meta = simulate_schrodinger(
            nx=nx, x_min=x_min, x_max=x_max, dt=dt, t_end=t_end, 
            snapshot_dt=snapshot_dt, kappa=kappa
        )
        
        trajs_real.append(traj_real)
        trajs_imag.append(traj_imag)
        meta0 = meta0 or meta
    
    trajs_real = torch.tensor(np.stack(trajs_real, axis=0), dtype=torch.float32)
    trajs_imag = torch.tensor(np.stack(trajs_imag, axis=0), dtype=torch.float32)
    
    train_real = trajs_real[:n_train]
    train_imag = trajs_imag[:n_train]
    test_real = trajs_real[n_train:]
    test_imag = trajs_imag[n_train:]
    
    meta = {
        "nx": nx, "x_min": x_min, "x_max": x_max, "dt": dt, 
        "t_end": t_end, "snapshot_dt": snapshot_dt, "kappa": kappa,
        "dx": meta0["dx"], "L": meta0["L"],
        "n_traj": n_traj, "n_train": n_train
    }
    
    result = {
        "train_real": train_real, "train_imag": train_imag,
        "test_real": test_real, "test_imag": test_imag,
        "meta": meta
    }
    
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/schrodinger_dataset.pt")
    p.add_argument("--n_traj", type=int, default=100)
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--x_min", type=float, default=-20.0)
    p.add_argument("--x_max", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--t_end", type=float, default=5.0)
    p.add_argument("--snapshot_dt", type=float, default=0.05)
    p.add_argument("--kappa", type=float, default=1.0)
    
    args = p.parse_args()
    
    ds = build_dataset(
        n_traj=args.n_traj, n_train=args.n_train,
        nx=args.nx, x_min=args.x_min, x_max=args.x_max,
        dt=args.dt, t_end=args.t_end, snapshot_dt=args.snapshot_dt,
        kappa=args.kappa
    )
    
    torch.save(ds, args.out)
    print(f"Saved dataset to {args.out}")
    print("Meta:", ds["meta"])


if __name__ == "__main__":
    main()
