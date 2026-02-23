import argparse
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm


def _gaussian_vortex(nx: int, ny: int, x0: float, y0: float, 
                     sigma: float, amp: float) -> np.ndarray:
    """
    生成单个高斯涡旋
    """
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    
    dx = np.minimum(np.abs(xx - x0), 1 - np.abs(xx - x0))
    dy = np.minimum(np.abs(yy - y0), 1 - np.abs(yy - y0))
    
    r2 = dx**2 + dy**2
    omega = amp * np.exp(-r2 / (2 * sigma**2))
    return omega


def _random_vorticity_field(nx: int, ny: int, n_vortices: int = 10,
                            amp_min: float = -5.0, amp_max: float = 5.0,
                            sigma_min: float = 0.05, sigma_max: float = 0.15) -> np.ndarray:
    """
    生成随机涡度场：多个高斯涡旋的叠加
    """
    omega = np.zeros((ny, nx))
    
    for _ in range(n_vortices):
        x0 = np.random.rand()
        y0 = np.random.rand()
        amp = np.random.uniform(amp_min, amp_max)
        sigma = np.random.uniform(sigma_min, sigma_max)
        omega += _gaussian_vortex(nx, ny, x0, y0, sigma, amp)
    
    return omega


def _fft_derivatives(omega_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用FFT计算导数（伪谱方法）
    """
    omega_x_hat = 1j * kx * omega_hat
    omega_y_hat = 1j * ky * omega_hat
    
    omega_xx_hat = -kx**2 * omega_hat
    omega_yy_hat = -ky**2 * omega_hat
    laplace_omega_hat = omega_xx_hat + omega_yy_hat
    
    return omega_x_hat, omega_y_hat, laplace_omega_hat


def _compute_velocity_from_vorticity(omega_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray, 
                                     dealias: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    从涡度计算速度场：ω = ∇×u
    使用流函数方法：∇²ψ = -ω，然后 u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    k_sq = kx**2 + ky**2
    k_sq[0, 0] = 1.0
    
    psi_hat = -omega_hat / k_sq
    psi_hat[0, 0] = 0.0
    
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    
    if dealias:
        u_hat = _dealias(u_hat)
        v_hat = _dealias(v_hat)
    
    u = np.fft.irfft2(u_hat)
    v = np.fft.irfft2(v_hat)
    
    return u, v


def _dealias(arr_hat: np.ndarray) -> np.ndarray:
    """
    2/3规则去混叠
    """
    ny, nx_half = arr_hat.shape
    nx = 2 * (nx_half - 1)
    
    kx_cut = int(nx / 3)
    ky_cut = int(ny / 3)
    
    arr_hat_dealias = arr_hat.copy()
    arr_hat_dealias[:, kx_cut:] = 0.0
    arr_hat_dealias[ky_cut:, :] = 0.0
    arr_hat_dealias[-ky_cut:, :] = 0.0
    
    return arr_hat_dealias


def _rhs_ns(omega: np.ndarray, kx: np.ndarray, ky: np.ndarray, nu: float) -> np.ndarray:
    """
    计算NS方程的右端项（涡度形式）
    ∂_t ω + u·∇ω = ν Δω
    """
    ny, nx = omega.shape
    
    omega_hat = np.fft.rfft2(omega)
    
    omega_x_hat, omega_y_hat, laplace_omega_hat = _fft_derivatives(omega_hat, kx, ky)
    
    omega_x = np.fft.irfft2(omega_x_hat)
    omega_y = np.fft.irfft2(omega_y_hat)
    laplace_omega = np.fft.irfft2(laplace_omega_hat)
    
    u, v = _compute_velocity_from_vorticity(omega_hat, kx, ky)
    
    advection = u * omega_x + v * omega_y
    diffusion = nu * laplace_omega
    
    rhs = -advection + diffusion
    
    return rhs


def _rk3_step(omega: np.ndarray, kx: np.ndarray, ky: np.ndarray, nu: float, dt: float) -> np.ndarray:
    """
    三阶Runge-Kutta时间积分
    """
    k1 = _rhs_ns(omega, kx, ky, nu)
    omega1 = omega + dt * k1
    
    k2 = _rhs_ns(omega1, kx, ky, nu)
    omega2 = 0.75 * omega + 0.25 * omega1 + 0.25 * dt * k2
    
    k3 = _rhs_ns(omega2, kx, ky, nu)
    omega_next = (1.0/3.0) * omega + (2.0/3.0) * omega2 + (2.0/3.0) * dt * k3
    
    return omega_next


def simulate_ns(
    nx: int = 64,
    ny: int = 64,
    dt: float = 1e-3,
    t_end: float = 2.0,
    snapshot_dt: float = 0.02,
    nu: float = 1e-3,
    n_vortices: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用伪谱方法模拟二维不可压缩Navier-Stokes方程（涡度形式）
    返回:
      traj_omega: [T, Ny, Nx]
      meta: 元数据
    """
    L = 1.0
    dx = L / nx
    dy = L / ny
    
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = kx[np.newaxis, :]
    ky = ky[:, np.newaxis]
    
    omega = _random_vorticity_field(nx, ny, n_vortices)
    
    n_steps = int(np.round(t_end / dt))
    snap_every = int(np.round(snapshot_dt / dt))
    assert snap_every >= 1
    
    snaps_omega = []
    t = 0.0
    
    for n in range(n_steps + 1):
        if n % snap_every == 0:
            snaps_omega.append(omega.copy())
        
        omega = _rk3_step(omega, kx, ky, nu, dt)
        
        t += dt
    
    traj_omega = np.stack(snaps_omega, axis=0)
    
    meta = {
        "L": L,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "nu": nu
    }
    
    return traj_omega, meta


def build_dataset(
    n_traj: int = 100,
    n_train: int = 80,
    nx: int = 64,
    ny: int = 64,
    dt: float = 1e-3,
    t_end: float = 2.0,
    snapshot_dt: float = 0.02,
    nu: float = 1e-3,
    n_vortices: int = 10,
) -> Dict[str, Any]:
    """
    构建NS数据集
    """
    trajs_omega = []
    meta0 = None
    
    for _ in tqdm(range(n_traj), desc="Simulating NS trajectories"):
        traj_omega, meta = simulate_ns(
            nx=nx, ny=ny, dt=dt, t_end=t_end, snapshot_dt=snapshot_dt,
            nu=nu, n_vortices=n_vortices
        )
        
        trajs_omega.append(traj_omega)
        meta0 = meta0 or meta
    
    trajs_omega = torch.tensor(np.stack(trajs_omega, axis=0), dtype=torch.float32)
    
    train_omega = trajs_omega[:n_train]
    test_omega = trajs_omega[n_train:]
    
    meta = {
        "nx": nx, "ny": ny, "dt": dt, "t_end": t_end, "snapshot_dt": snapshot_dt,
        "nu": nu, "dx": meta0["dx"], "dy": meta0["dy"], "L": meta0["L"],
        "n_traj": n_traj, "n_train": n_train
    }
    
    result = {
        "train_omega": train_omega,
        "test_omega": test_omega,
        "meta": meta
    }
    
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/ns_dataset.pt")
    p.add_argument("--n_traj", type=int, default=100)
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--t_end", type=float, default=2.0)
    p.add_argument("--snapshot_dt", type=float, default=0.02)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--n_vortices", type=int, default=10)
    
    args = p.parse_args()
    
    ds = build_dataset(
        n_traj=args.n_traj, n_train=args.n_train,
        nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.t_end, 
        snapshot_dt=args.snapshot_dt, nu=args.nu, n_vortices=args.n_vortices
    )
    
    torch.save(ds, args.out)
    print(f"Saved dataset to {args.out}")
    print("Meta:", ds["meta"])


if __name__ == "__main__":
    main()
