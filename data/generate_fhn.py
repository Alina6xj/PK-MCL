import argparse
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm


def _ddx_2d(u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算2D空间导数（Neumann边界条件）
    u: [Ny, Nx]
    returns: (du_dx, du_dy)
    """
    Ny, Nx = u.shape
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
    
    du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    du_dy[0, :] = (u[1, :] - u[0, :]) / dx
    du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dx
    
    return du_dx, du_dy


def _ddxx_2d(u: np.ndarray, dx: float) -> np.ndarray:
    """
    计算2D拉普拉斯算子（Neumann边界条件）
    u: [Ny, Nx]
    returns: d^2u/dx^2 + d^2u/dy^2
    """
    Ny, Nx = u.shape
    laplacian = np.zeros_like(u)
    
    laplacian[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    laplacian[1:-1, :] += (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dx * dx)
    
    laplacian[:, 0] = (u[:, 2] - 2 * u[:, 1] + u[:, 0]) / (dx * dx)
    laplacian[:, -1] = (u[:, -1] - 2 * u[:, -2] + u[:, -3]) / (dx * dx)
    laplacian[0, :] += (u[2, :] - 2 * u[1, :] + u[0, :]) / (dx * dx)
    laplacian[-1, :] += (u[-1, :] - 2 * u[-2, :] + u[-3, :]) / (dx * dx)
    
    return laplacian


def _gaussian_initial(nx: int, ny: int, dx: float, A: float = 0.05, 
                     x0: float = 0.5, y0: float = 0.5, sigma: float = 0.1) -> np.ndarray:
    """
    生成高斯扰动的初始条件
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    
    v0 = A * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
    return v0


def _fhn_nonlinear(v: np.ndarray, w: np.ndarray, a: float = 0.7, 
                   b: float = 0.8, I: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    FHN系统的非线性项
    f(v, w) = v - v^3/3 - w
    g(v, w) = v + a - b w
    """
    f = v - (v ** 3) / 3 - w
    g = v + a - b * w
    return f, g


def _rhs_fhn(v: np.ndarray, w: np.ndarray, dx: float, D_v: float = 0.1, 
             D_w: float = 0.05, eps: float = 0.08, a: float = 0.7, 
             b: float = 0.8, I: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算FHN系统的右端项
    """
    laplace_v = _ddxx_2d(v, dx)
    laplace_w = _ddxx_2d(w, dx)
    f, g = _fhn_nonlinear(v, w, a, b, I)
    
    dv_dt = D_v * laplace_v + f + I
    dw_dt = D_w * laplace_w + eps * g
    
    return dv_dt, dw_dt


def simulate_fhn(
    nx: int = 100,
    ny: int = 100,
    dt: float = 1.0e-6,
    t_end: float = 1.0,
    snapshot_dt: float = 0.01,
    D_v: float = 0.1,
    D_w: float = 0.05,
    eps: float = 0.08,
    a: float = 0.7,
    b: float = 0.8,
    I: float = 0.5,
    A: float = 0.05,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用RK2积分方法模拟FHN系统，提高数值稳定性
    返回:
      traj_v: [T, Ny, Nx]
      traj_w: [T, Ny, Nx]
      meta: 元数据
    """
    L = 1.0
    dx = L / nx
    dy = L / ny
    
    v = _gaussian_initial(nx, ny, dx, A, x0, y0, sigma)
    w = np.zeros_like(v)
    
    n_steps = int(np.round(t_end / dt))
    snap_every = int(np.round(snapshot_dt / dt))
    assert snap_every >= 1
    
    snaps_v = []
    snaps_w = []
    t = 0.0
    
    for n in range(n_steps + 1):
        if n % snap_every == 0:
            snaps_v.append(v.copy())
            snaps_w.append(w.copy())
        
        dv_dt1, dw_dt1 = _rhs_fhn(v, w, dx, D_v, D_w, eps, a, b, I)
        
        v1 = v + dt * dv_dt1
        w1 = w + dt * dw_dt1
        
        dv_dt2, dw_dt2 = _rhs_fhn(v1, w1, dx, D_v, D_w, eps, a, b, I)
        
        v = v + 0.5 * dt * (dv_dt1 + dv_dt2)
        w = w + 0.5 * dt * (dw_dt1 + dw_dt2)
        
        t += dt
    
    traj_v = np.stack(snaps_v, axis=0)
    traj_w = np.stack(snaps_w, axis=0)
    
    meta = {
        "L": L,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "D_v": D_v,
        "D_w": D_w,
        "eps": eps,
        "a": a,
        "b": b,
        "I": I
    }
    
    return traj_v, traj_w, meta


def build_dataset(
    n_traj: int = 10,
    n_train: int = 8,
    nx: int = 100,
    ny: int = 100,
    dt: float = 1.0e-6,
    t_end: float = 1.0,
    snapshot_dt: float = 0.01,
    D_v: float = 0.1,
    D_w: float = 0.05,
    eps: float = 0.08,
    a: float = 0.7,
    b: float = 0.8,
    I: float = 0.5,
    A: float = 0.05,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.1,
) -> Dict[str, Any]:
    """
    构建FHN数据集
    """
    trajs_v = []
    trajs_w = []
    meta0 = None
    
    for _ in tqdm(range(n_traj), desc="Simulating FHN trajectories"):
        traj_v, traj_w, meta = simulate_fhn(
            nx=nx, ny=ny, dt=dt, t_end=t_end, snapshot_dt=snapshot_dt,
            D_v=D_v, D_w=D_w, eps=eps, a=a, b=b, I=I,
            A=A, x0=x0, y0=y0, sigma=sigma
        )
        
        trajs_v.append(traj_v)
        trajs_w.append(traj_w)
        meta0 = meta0 or meta
    
    trajs_v = torch.tensor(np.stack(trajs_v, axis=0), dtype=torch.float32)
    trajs_w = torch.tensor(np.stack(trajs_w, axis=0), dtype=torch.float32)
    
    train_v = trajs_v[:n_train]
    train_w = trajs_w[:n_train]
    test_v = trajs_v[n_train:]
    test_w = trajs_w[n_train:]
    
    meta = {
        "nx": nx, "ny": ny, "dt": dt, "t_end": t_end, "snapshot_dt": snapshot_dt,
        "D_v": D_v, "D_w": D_w, "eps": eps, "a": a, "b": b, "I": I,
        "dx": meta0["dx"], "dy": meta0["dy"], "L": meta0["L"],
        "n_traj": n_traj, "n_train": n_train
    }
    
    result = {
        "train_v": train_v, "train_w": train_w,
        "test_v": test_v, "test_w": test_w,
        "meta": meta
    }
    
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/fhn_dataset.pt")
    p.add_argument("--n_traj", type=int, default=10)
    p.add_argument("--n_train", type=int, default=8)
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--dt", type=float, default=1.0e-6)
    p.add_argument("--t_end", type=float, default=1.0)
    p.add_argument("--snapshot_dt", type=float, default=0.01)
    p.add_argument("--D_v", type=float, default=0.1)
    p.add_argument("--D_w", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.08)
    p.add_argument("--a", type=float, default=0.7)
    p.add_argument("--b", type=float, default=0.8)
    p.add_argument("--I", type=float, default=0.5)
    p.add_argument("--A", type=float, default=0.05)
    p.add_argument("--x0", type=float, default=0.5)
    p.add_argument("--y0", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.1)
    
    args = p.parse_args()
    
    ds = build_dataset(
        n_traj=args.n_traj, n_train=args.n_train,
        nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.t_end, 
        snapshot_dt=args.snapshot_dt,
        D_v=args.D_v, D_w=args.D_w, eps=args.eps, a=args.a, b=args.b, I=args.I,
        A=args.A, x0=args.x0, y0=args.y0, sigma=args.sigma
    )
    
    torch.save(ds, args.out)
    print(f"Saved dataset to {args.out}")
    print("Meta:", ds["meta"])


if __name__ == "__main__":
    main()
