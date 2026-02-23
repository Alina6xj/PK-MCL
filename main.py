import argparse
from pathlib import Path

import torch

from utils.config import load_yaml
from utils.seed import set_seed
from utils.io import ensure_dir

def _maybe_generate_data(cfg):
    data_cfg = cfg["data"]
    dataset_path = data_cfg["dataset_path"]
    if Path(dataset_path).exists():
        return
    
    system_type = cfg.get("system", "burgers")
    
    if system_type == "fhn":
        from data.generate_fhn import build_dataset
        ds = build_dataset(
            n_traj=int(data_cfg["n_traj"]),
            n_train=int(data_cfg["n_train"]),
            nx=int(data_cfg["nx"]),
            ny=int(data_cfg.get("ny", 100)),
            dt=float(data_cfg["dt"]),
            t_end=float(data_cfg["t_end"]),
            snapshot_dt=float(data_cfg["snapshot_dt"]),
            D_v=float(data_cfg["D_v"]),
            D_w=float(data_cfg["D_w"]),
            eps=float(data_cfg["eps"]),
            a=float(data_cfg["a"]),
            b=float(data_cfg["b"]),
            I=float(data_cfg["I"]),
            A=float(data_cfg["A"]),
            x0=float(data_cfg["x0"]),
            y0=float(data_cfg["y0"]),
            sigma=float(data_cfg["sigma"]),
        )
    elif system_type == "ns":
        from data.generate_ns import build_dataset
        ds = build_dataset(
            n_traj=int(data_cfg["n_traj"]),
            n_train=int(data_cfg["n_train"]),
            nx=int(data_cfg["nx"]),
            ny=int(data_cfg.get("ny", 64)),
            dt=float(data_cfg["dt"]),
            t_end=float(data_cfg["t_end"]),
            snapshot_dt=float(data_cfg["snapshot_dt"]),
            nu=float(data_cfg["nu"]),
            n_vortices=int(data_cfg.get("n_vortices", 10)),
        )
    elif system_type == "schrodinger":
        from data.generate_schrodinger import build_dataset
        ds = build_dataset(
            n_traj=int(data_cfg["n_traj"]),
            n_train=int(data_cfg["n_train"]),
            nx=int(data_cfg["nx"]),
            x_min=float(data_cfg["x_min"]),
            x_max=float(data_cfg["x_max"]),
            dt=float(data_cfg["dt"]),
            t_end=float(data_cfg["t_end"]),
            snapshot_dt=float(data_cfg["snapshot_dt"]),
            kappa=float(data_cfg["kappa"]),
        )
    elif system_type == "rd":
        from data.generate_rd import build_dataset
        ds = build_dataset(
            n_traj=int(data_cfg["n_traj"]),
            n_train=int(data_cfg["n_train"]),
            nx=int(data_cfg["nx"]),
            ny=int(data_cfg.get("ny", 100)),
            dt=float(data_cfg["dt"]),
            t_end=float(data_cfg["t_end"]),
            snapshot_dt=float(data_cfg["snapshot_dt"]),
            D_u=float(data_cfg["D_u"]),
            D_v=float(data_cfg["D_v"]),
            F=float(data_cfg["F"]),
            k=float(data_cfg["k"]),
            square_size=int(data_cfg.get("square_size", 10)),
        )
    else:
        from data.generate_burgers import build_dataset, AugmentConfig
        augment_cfg = None
        if "augmentation" in data_cfg and data_cfg["augmentation"].get("enabled", False):
            aug_cfg_dict = data_cfg["augmentation"]
            augment_cfg = AugmentConfig(
                num_ops_min=int(aug_cfg_dict.get("num_ops_min", 1)),
                num_ops_max=int(aug_cfg_dict.get("num_ops_max", 3)),
                noise_std_min=float(aug_cfg_dict.get("noise_std_min", 0.0)),
                noise_std_max=float(aug_cfg_dict.get("noise_std_max", 0.02)),
                mask_ratio_min=float(aug_cfg_dict.get("mask_ratio_min", 0.0)),
                mask_ratio_max=float(aug_cfg_dict.get("mask_ratio_max", 0.15)),
                mask_block_min=int(aug_cfg_dict.get("mask_block_min", 8)),
                mask_block_max=int(aug_cfg_dict.get("mask_block_max", 32)),
                scale_min=float(aug_cfg_dict.get("scale_min", 0.9)),
                scale_max=float(aug_cfg_dict.get("scale_max", 1.1)),
                bias_min=float(aug_cfg_dict.get("bias_min", -0.05)),
                bias_max=float(aug_cfg_dict.get("bias_max", 0.05)),
                k_cut_min=int(aug_cfg_dict.get("k_cut_min", 8)),
                k_cut_max=int(aug_cfg_dict.get("k_cut_max", 64)),
                augment_control=bool(aug_cfg_dict.get("augment_control", False)),
                control_noise_std=float(aug_cfg_dict.get("control_noise_std", 0.005)),
            )
        
        ds = build_dataset(
            n_traj=int(data_cfg["n_traj"]),
            n_train=int(data_cfg["n_train"]),
            nx=int(data_cfg["nx"]),
            dt=float(data_cfg["dt"]),
            t_end=float(data_cfg["t_end"]),
            snapshot_dt=float(data_cfg["snapshot_dt"]),
            nu=float(data_cfg["nu"]),
            forcing=bool(data_cfg["forcing"]),
            forcing_amp=float(data_cfg["forcing_amp"]),
            forcing_n_gauss=int(data_cfg["forcing_n_gauss"]),
            ic_n_modes=int(data_cfg["ic_n_modes"]),
            ic_amp=float(data_cfg["ic_amp"]),
            augment_cfg=augment_cfg,
        )
    
    ensure_dir(str(Path(dataset_path).parent))
    torch.save(ds, dataset_path)
    print(f"[main] Generated dataset at {dataset_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    p.add_argument("--config", type=str, default="configs/burgers.yaml")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path for test; default uses cfg.io")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    _maybe_generate_data(cfg)
    data_cfg = cfg["data"]
    dataset = torch.load(data_cfg["dataset_path"], map_location="cpu", weights_only=False)

    io_cfg = cfg["io"]
    ckpt_dir = io_cfg["ckpt_dir"]
    out_dir = io_cfg["out_dir"]
    ckpt_name = io_cfg["ckpt_name"]

    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)

    if args.mode == "train":
        from train.train import train
        ckpt_path = str(Path(ckpt_dir) / ckpt_name)
        train(cfg, dataset, ckpt_path=ckpt_path)
    else:
        from test.eval import evaluate
        ckpt_path = args.ckpt or str(Path(ckpt_dir) / ckpt_name)
        evaluate(cfg, dataset, ckpt_path=ckpt_path, out_dir=out_dir)

if __name__ == "__main__":
    main()
