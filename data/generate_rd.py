import argparse
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm


def _ddx_2d_periodic(u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算2D空间导数（周期边界条件）
    u: [Ny, Nx]
    returns: (du_dx, du_dy)
    """
    Ny, Nx = u.shape
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    
    return du_dx, du_dy


def _ddxx_2d_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """
    计算2D拉普拉斯算子（周期边界条件）
    u: [Ny, Nx]
    returns: d^2u/dx^2 + d^2u/dy^2
    """
    Ny, Nx = u.shape
    laplacian = np.zeros_like(u)
    
    laplacian = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (dx * dx)
    laplacian += (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (dx * dx)
    
    return laplacian


def _random_square_perturbation(nx: int, ny: int, u0: float = 1.0, v0: float = 0.0, 
                                square_size: int = 10, amp: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成带有小随机方形扰动的初始条件
    u ≈ 1, v ≈ 0 with small random square perturbation
    """
    u = np.ones((ny, nx)) * u0
    v = np.ones((ny, nx)) * v0
    
    cx = np.random.randint(square_size, nx - square_size)
    cy = np.random.randint(square_size, ny - square_size)
    
    x_start = cx - square_size // 2
    x_end = cx + square_size // 2
    y_start = cy - square_size // 2
    y_end = cy + square_size // 2
    
    u[y_start:y_end, x_start:x_end] -= amp * np.random.rand(square_size, square_size)
    v[y_start:y_end, x_start:x_end] += amp * np.random.rand(square_size, square_size)
    
    return u, v


def _rhs_rd(u: np.ndarray, v: np.ndarray, dx: float, D_u: float, D_v: float, 
           F: float, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算RD系统的右端项
    ∂_t u = D_u ∇²u - u v² + F (1 - u)
    ∂_t v = D_v ∇²v + u v² - (F + k) v
    """
    laplace_u = _ddxx_2d_periodic(u, dx)
    laplace_v = _ddxx_2d_periodic(v, dx)
    
    uv2 = u * v ** 2
    
    du_dt = D_u * laplace_u - uv2 + F * (1 - u)
    dv_dt = D_v * laplace_v + uv2 - (F + k) * v
    
    return du_dt, dv_dt


def simulate_rd(
    nx: int = 100,
    ny: int = 100,
    dt: float = 1.0,
    t_end: float = 500.0,
    snapshot_dt: float = 5.0,
    D_u: float = 2e-5,
    D_v: float = 1e-5,
    F: float = 0.035,
    k: float = 0.065,
    square_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用显式有限差分法模拟RD（反应扩散）系统
    返回:
      traj_u: [T, Ny, Nx]
      traj_v: [T, Ny, Nx]
      meta: 元数据
    """
    L = 1.0
    dx = L / nx
    dy = L / ny
    
    u, v = _random_square_perturbation(nx, ny, square_size=square_size)
    
    n_steps = int(np.round(t_end / dt))
    snap_every = int(np.round(snapshot_dt / dt))
    assert snap_every >= 1
    
    snaps_u = []
    snaps_v = []
    t = 0.0
    
    for n in range(n_steps + 1):
        if n % snap_every == 0:
            snaps_u.append(u.copy())
            snaps_v.append(v.copy())
        
        du_dt, dv_dt = _rhs_rd(u, v, dx, D_u, D_v, F, k)
        
        u = u + dt * du_dt
        v = v + dt * dv_dt
        
        t += dt
    
    traj_u = np.stack(snaps_u, axis=0)
    traj_v = np.stack(snaps_v, axis=0)
    
    meta = {
        "L": L,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "D_u": D_u,
        "D_v": D_v,
        "F": F,
        "k": k
    }
    
    return traj_u, traj_v, meta


def build_dataset(
    n_traj: int = 100,
    n_train: int = 80,
    nx: int = 100,
    ny: int = 100,
    dt: float = 1.0,
    t_end: float = 500.0,
    snapshot_dt: float = 5.0,
    D_u: float = 2e-5,
    D_v: float = 1e-5,
    F: float = 0.035,
    k: float = 0.065,
    square_size: int = 10,
) -> Dict[str, Any]:
    """
    构建RD数据集
    """
    trajs_u = []
    trajs_v = []
    meta0 = None
    
    for _ in tqdm(range(n_traj), desc="Simulating RD trajectories"):
        traj_u, traj_v, meta = simulate_rd(
            nx=nx, ny=ny, dt=dt, t_end=t_end, snapshot_dt=snapshot_dt,
            D_u=D_u, D_v=D_v, F=F, k=k, square_size=square_size
        )
        
        trajs_u.append(traj_u)
        trajs_v.append(traj_v)
        meta0 = meta0 or meta
    
    trajs_u = torch.tensor(np.stack(trajs_u, axis=0), dtype=torch.float32)
    trajs_v = torch.tensor(np.stack(trajs_v, axis=0), dtype=torch.float32)
    
    train_u = trajs_u[:n_train]
    train_v = trajs_v[:n_train]
    test_u = trajs_u[n_train:]
    test_v = trajs_v[n_train:]
    
    meta = {
        "nx": nx, "ny": ny, "dt": dt, "t_end": t_end, "snapshot_dt": snapshot_dt,
        "D_u": D_u, "D_v": D_v, "F": F, "k": k,
        "dx": meta0["dx"], "dy": meta0["dy"], "L": meta0["L"],
        "n_traj": n_traj, "n_train": n_train
    }
    
    result = {
        "train_u": train_u, "train_v": train_v,
        "test_u": test_u, "test_v": test_v,
        "meta": meta
    }
    
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/rd_dataset.pt")
    p.add_argument("--n_traj", type=int, default=100)
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--t_end", type=float, default=500.0)
    p.add_argument("--snapshot_dt", type=float, default=5.0)
    p.add_argument("--D_u", type=float, default=2e-5)
    p.add_argument("--D_v", type=float, default=1e-5)
    p.add_argument("--F", type=float, default=0.035)
    p.add_argument("--k", type=float, default=0.065)
    p.add_argument("--square_size", type=int, default=10)
    
    args = p.parse_args()
    
    ds = build_dataset(
        n_traj=args.n_traj, n_train=args.n_train,
        nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.t_end, 
        snapshot_dt=args.snapshot_dt, D_u=args.D_u, D_v=args.D_v,
        F=args.F, k=args.k, square_size=args.square_size
    )
    
    torch.save(ds, args.out)
    print(f"Saved dataset to {args.out}")
    print("Meta:", ds["meta"])


if __name__ == "__main__":
    main()
