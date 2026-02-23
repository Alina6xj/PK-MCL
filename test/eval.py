from typing import Dict, Any, Optional
import os

import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from models.pkmcl import PKMCL, PKMCLConfig
from models.physics_library import LibraryConfig
from utils.io import ensure_dir, load_ckpt

class BurgersDataset(Dataset):
    def __init__(self, u: torch.Tensor, F: torch.Tensor):
        self.u = u
        self.F = F

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.F[idx]

def _make_model_from_cfg(cfg: Dict[str, Any], meta: Dict[str, Any]) -> PKMCL:
    mcfg = cfg["model"]
    lcfg = cfg["model"]["library"]
    lib_cfg = LibraryConfig(**lcfg)
    from models.pkmcl import PKMCLConfig
    ms = mcfg.get("multiscale", {}) or {}
    pkmcl_cfg = PKMCLConfig(
        nx=int(meta["nx"]),
        latent_dim=int(mcfg["latent_dim"]),
        enc_channels=int(mcfg["enc_channels"]),
        dec_channels=int(mcfg["dec_channels"]),
        use_forcing=bool(mcfg["use_forcing"]),
        lib_cfg=lib_cfg,

        multiscale=bool(ms.get("enabled", False)),
        k_low=int(ms.get("k_low", 12)),
        k_mid=int(ms.get("k_mid", 48)),
        latent_dim_low=int(ms.get("latent_dim_low", int(mcfg["latent_dim"]))),
        latent_dim_mid=int(ms.get("latent_dim_mid", int(mcfg["latent_dim"]))),
        high_channels=int(ms.get("high_channels", 64)),
    )
    return PKMCL(pkmcl_cfg)

@torch.no_grad()
def evaluate(cfg: Dict[str, Any], dataset: Dict[str, Any], ckpt_path: str, out_dir: str):
    ensure_dir(out_dir)
    train_cfg = cfg["train"]
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    ckpt = load_ckpt(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    model_cfg = ckpt["config"]
    model = _make_model_from_cfg(model_cfg, meta).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 获取dx和测试数据的一个样本
    dx = float(meta["dx"])
    test_u = dataset["test_u"]
    test_F = dataset["test_F"]
    
    # 使用测试数据的第一个样本
    u_sample = torch.from_numpy(test_u[0:1])  # [1, T, Nx]
    u_sample = u_sample.to(device)
    
    print("Loaded equation:", ckpt.get("equation", model.discovered_equation(u_sample, dx)))

    ds = BurgersDataset(test_u, test_F)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # 取一个样本做可视化
    u, Fsrc = next(iter(dl))
    u = u.to(device)     # [1, T, Nx]
    Fsrc = Fsrc.to(device)
    u0 = u[:, 0, :]
    T = u.shape[1] - 1   # 可预测长度（对齐真值的未来步数）

    u_pred = model.rollout(u0, steps=T, Fsrc=Fsrc if model.cfg.use_forcing else None)  # [1, T, Nx]
    u_true = u[:, 1:, :]  # [1, T, Nx]

    mse = torch.mean((u_pred - u_true) ** 2).item()
    rmse = mse ** 0.5
    print(f"Test sample RMSE: {rmse:.6f}")

    # 绘制时空图
    u_true_np = u_true.squeeze(0).detach().cpu().numpy()  # [T, Nx]
    u_pred_np = u_pred.squeeze(0).detach().cpu().numpy()

    def plot_spacetime(U, title, fname):
        plt.figure()
        plt.imshow(U.T, origin="lower", aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("time index")
        plt.ylabel("x index")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    plot_spacetime(u_true_np, "True u(x,t)", "rollout_true.png")
    plot_spacetime(u_pred_np, "Predicted u(x,t) (PK-MCL)", "rollout_pred.png")
    plot_spacetime((u_pred_np - u_true_np), "Error u_pred - u_true", "rollout_err.png")

    # 截面曲线对比（选 3 个时刻）
    import numpy as np
    t_ids = [0, T//2, T-1]
    x = np.arange(u_true_np.shape[1])

    plt.figure()
    for tid in t_ids:
        plt.plot(x, u_true_np[tid], label=f"true t={tid}")
        plt.plot(x, u_pred_np[tid], "--", label=f"pred t={tid}")
    plt.legend(fontsize=8, ncol=2)
    plt.title("Slices u(x) at selected times")
    plt.xlabel("x index")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "slices.png"), dpi=200)
    plt.close()

    print(f"Saved figures to {out_dir}")
