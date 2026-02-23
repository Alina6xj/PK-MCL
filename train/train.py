from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.pkmcl import PKMCL, PKMCLConfig
from models.physics_library import LibraryConfig
from utils.io import save_ckpt, ensure_dir
from utils.diff import dt_central, time_smoothing

class BurgersDataset(Dataset):
    def __init__(self, u: torch.Tensor, F: torch.Tensor, view1: Optional[torch.Tensor] = None, view2: Optional[torch.Tensor] = None):
        self.u = u  # [N, T, Nx]
        self.F = F  # [N, Nx]
        self.view1 = view1  # [N, T, Nx] (augmented view 1)
        self.view2 = view2  # [N, T, Nx] (augmented view 2)
        self.has_views = view1 is not None and view2 is not None

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        if self.has_views:
            return self.u[idx], self.F[idx], self.view1[idx], self.view2[idx]
        else:
            return self.u[idx], self.F[idx]

def _make_model(cfg: Dict[str, Any], meta: Dict[str, Any]) -> PKMCL:
    mcfg = cfg["model"]
    lcfg = cfg["model"]["library"]
    lib_cfg = LibraryConfig(**lcfg)
    ms = mcfg.get("multiscale", {}) or {}
    pkmcl_cfg = PKMCLConfig(
        nx=int(meta["nx"]),
        latent_dim=int(mcfg["latent_dim"]),
        enc_channels=int(mcfg["enc_channels"]),
        dec_channels=int(mcfg["dec_channels"]),
        use_forcing=bool(mcfg["use_forcing"]),
        lib_cfg=lib_cfg,

        # multi-scale (optional)
        multiscale=bool(ms.get("enabled", False)),
        k_low=int(ms.get("k_low", 12)),
        k_mid=int(ms.get("k_mid", 48)),
        latent_dim_low=int(ms.get("latent_dim_low", int(mcfg["latent_dim"]))),
        latent_dim_mid=int(ms.get("latent_dim_mid", int(mcfg["latent_dim"]))),
        high_channels=int(ms.get("high_channels", 64)),
    )
    return PKMCL(pkmcl_cfg)

def train(cfg: Dict[str, Any], dataset: Dict[str, Any], ckpt_path: str) -> None:
    train_cfg = cfg["train"]
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    meta = dataset["meta"]
    dt = float(meta["snapshot_dt"])  # 注意：数据保存的是每 snapshot_dt 一个快照
    dx = float(meta["dx"])

    model = _make_model(cfg, meta).to(device)

    if bool(train_cfg.get("use_consistency", False)):
        model.build_target()

    opt = torch.optim.Adam(
        list(model.dyn.parameters()) + list(model.phys.parameters()),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0))
    )

    # 检查数据集是否包含增强视图
    has_views = "train_view1" in dataset and "train_view2" in dataset
    if has_views:
        ds = BurgersDataset(
            dataset["train_u"], 
            dataset["train_F"],
            dataset["train_view1"],
            dataset["train_view2"]
        )
    else:
        ds = BurgersDataset(dataset["train_u"], dataset["train_F"])
    dl = DataLoader(ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg["num_workers"]))

    # 三阶段训练配置（向后兼容旧的两阶段配置）
    epochs_phaseA = int(train_cfg.get("epochs_phaseA", 0))
    epochs_phaseB = int(train_cfg.get("epochs_phaseB", 0))
    epochs_phaseC = int(train_cfg.get("epochs_phaseC", 0))
    
    # 向后兼容：如果没有设置三阶段参数，则使用旧的两阶段参数
    if epochs_phaseA == 0 and epochs_phaseB == 0:
        epochs_phaseA = int(train_cfg.get("epochs_phase1", 0))
        epochs_phaseB = int(train_cfg.get("epochs_phase2", 0))
        print(f"[Backward Compatibility] Using old phase1/phase2 as phaseA/phaseB: phaseA={epochs_phaseA}, phaseB={epochs_phaseB}")
    
    # 预测参数
    rollout_steps = int(train_cfg["rollout_steps"])
    
    # 损失权重配置
    lambda_phy = float(train_cfg["lambda_phy"])
    lambda_l1 = float(train_cfg["lambda_l1"])
    use_con = bool(train_cfg.get("use_consistency", False))
    
    # 各阶段特定参数（向后兼容旧的配置）
    # 检查是否使用了旧的单一一致性权重参数
    has_old_con = "lambda_con" in train_cfg and "lambda_con_A" not in train_cfg
    
    lambda_con_A = float(train_cfg.get("lambda_con_A", 1.0))  # Phase A：较大的一致性权重
    lambda_con_B = float(train_cfg.get("lambda_con_B", 0.0))  # Phase B：较小或为0的一致性权重
    lambda_con_C = float(train_cfg.get("lambda_con_C", 0.1))  # Phase C：中等的一致性权重
    
    # 向后兼容：如果存在旧的lambda_con参数且没有新参数，则使用旧参数
    if has_old_con:
        old_lambda_con = float(train_cfg["lambda_con"])
        lambda_con_A = old_lambda_con
        lambda_con_B = old_lambda_con * 0.01  # Phase B使用较小的一致性权重
        lambda_con_C = old_lambda_con * 0.1   # Phase C使用中等的一致性权重
        print(f"[Backward Compatibility] Using old lambda_con={old_lambda_con} with phase-specific scaling")
    
    alpha_C = float(train_cfg.get("alpha_C", 0.1))  # Phase C：物理约束权重
    
    # EMA更新参数
    ema_tau = float(train_cfg.get("ema_tau", 0.99))

    def l1_penalty():
        # 计算全局ξ参数的L1正则化
        if hasattr(model.phys, 'get_l1_regularization'):
            return model.phys.get_l1_regularization(lambda_l1)
        else:
            return torch.tensor(0.0, device=device)

    def log_training_info(u_true, Theta=None, u_t_pred=None):
        """
        打印训练信息：ξ稀疏度、关键项比值、物理残差分布
        """
        with torch.no_grad():
            # 获取当前的ξ参数（投影后）
            xi_proj = model.phys._project_xi()
            xi_list = xi_proj.detach().cpu().tolist()
            
            # 1. ξ的稀疏度：nnz(|xi|>thr)
            thr = 1e-3
            nnz = sum(1 for x in xi_list if abs(x) > thr)
            sparsity = nnz / len(xi_list)
            
            # 2. Burgers关键项比值：|xi_uux| 与 |xi_uxx| 的稳定性
            term_names = model.phys.library.term_names()
            xi_dict = dict(zip(term_names, xi_list))
            
            ratio = 0.0
            if 'u * u_x' in xi_dict and 'u_xx' in xi_dict:
                xi_uux = abs(xi_dict['u * u_x'])
                xi_uxx = abs(xi_dict['u_xx'])
                if xi_uxx > thr:
                    ratio = xi_uux / xi_uxx
            
            # 3. 物理残差分布：||u_t - Θξ|| 的均值/分位数
            residual_stats = None
            if Theta is not None and u_t_pred is not None:
                # 计算物理残差
                u_t_true = model.phys.library(u_true, dx)[0] @ xi_proj
                residual = u_t_true - u_t_pred
                residual_mean = residual.abs().mean().item()
                residual_quantile = torch.quantile(residual.abs(), 0.95).item()
                residual_stats = (residual_mean, residual_quantile)
            
            # 打印信息
            info = f"[Training Info] "
            info += f"Sparsity: {sparsity:.2f} (nnz={nnz}/{len(xi_list)}), "
            info += f"Burgers Ratio (u*u_x/uxx): {ratio:.3f}, "
            if residual_stats:
                info += f"Phys Residual (mean/95%): {residual_stats[0]:.4f}/{residual_stats[1]:.4f}"
            
            print(info)

    # -------- Phase A: 表示与演化预训练 --------
    # 目的：先让多尺度 Koopman + 高频补偿学到稳定的可滚动预测与表征
    # 更新参数：θ_dyn（encoder/koopman/residual/decoder 等）
    # 冻结：ξ.requires_grad = False
    if epochs_phaseA > 0:
        print("===== Starting Phase A: Representation and Evolution Pretraining =====")
        
        # 冻结ξ参数
        if hasattr(model.phys, 'xi'):
            model.phys.xi.requires_grad = False
        
        # 创建只包含动态参数的优化器
        dyn_params = list(model.dyn.parameters())
        opt_A = torch.optim.Adam(
            dyn_params,
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0))
        )
        
        for epoch in range(1, epochs_phaseA + 1):
            model.train()
            pbar = tqdm(dl, desc=f"[PhaseA] Epoch {epoch}/{epochs_phaseA}")
            for batch in pbar:
                # 处理批次数据，支持有无视图的情况
                if len(batch) == 2:
                    u, Fsrc = batch
                    u_view1 = u_view2 = None
                elif len(batch) == 4:
                    u, Fsrc, u_view1, u_view2 = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                
                u = u.to(device)     # [B, T, Nx]
                Fsrc = Fsrc.to(device)
                u0 = u[:, 0, :]
                u_true = u[:, 1:1+rollout_steps, :]  # 监督未来 rollout_steps 步
                
                # 处理视图数据
                if u_view1 is not None and u_view2 is not None:
                    u_view1 = u_view1.to(device)
                    u_view2 = u_view2.to(device)
                    view1_0 = u_view1[:, 0, :]  # [B, Nx] 初始状态
                    view2_0 = u_view2[:, 0, :]  # [B, Nx] 初始状态

                u_pred = model.rollout(u0, steps=rollout_steps, Fsrc=Fsrc if model.cfg.use_forcing else None)
                loss = model.loss_rec(u_pred, u_true)

                if use_con:
                    if u_view1 is not None and u_view2 is not None:
                        # 使用预增强的视图计算一致性损失
                        loss = loss + lambda_con_A * model.loss_con(
                            u0, Fsrc=Fsrc, u_view1=view1_0, u_view2=view2_0
                        )
                    else:
                        # 使用原始的高斯噪声增强
                        loss = loss + lambda_con_A * model.loss_con(u0, Fsrc=Fsrc)
                
                opt_A.zero_grad()
                loss.backward()
                opt_A.step()

                if use_con:
                    model.ema_update(ema_tau)

                pbar.set_postfix({"loss": float(loss.item())})

    # -------- Phase B: 结构参数识别 --------
    # 目的：在“稳定表示”下，让 ξ 收敛出稀疏且正确的方程
    # 方式：冻结 θ_dyn，只更新 ξ
    # 损失：L = L_phy(ξ) + λ∥ξ∥₁
    if epochs_phaseB > 0:
        print("===== Starting Phase B: Structure Parameter Identification =====")
        
        # 冻结动态参数
        for param in model.dyn.parameters():
            param.requires_grad = False
        
        # 解冻ξ相关参数
        if hasattr(model.phys, 'xi'):
            model.phys.xi.requires_grad = True
        if hasattr(model.phys, 'gate_logits'):
            model.phys.gate_logits.requires_grad = True
        
        # 创建只包含ξ参数的优化器
        xi_params = []
        if hasattr(model.phys, 'xi'):
            xi_params.append(model.phys.xi)
        if hasattr(model.phys, 'gate_logits'):
            xi_params.append(model.phys.gate_logits)
        
        # 增加学习率，使Phase B的参数更新更有效
        phaseB_lr = float(train_cfg["lr"]) * 1.0  # 增加到基础学习率的1.0倍
        opt_B = torch.optim.Adam(
            xi_params,
            lr=phaseB_lr,  # 使用较高的学习率以加快收敛
            weight_decay=0.0  # 对ξ不使用权重衰减
        )
        
        # 添加学习率调度器，在训练过程中逐渐降低学习率
        scheduler_B = torch.optim.lr_scheduler.ExponentialLR(
            opt_B,
            gamma=0.99  # 每个epoch学习率乘以0.99
        )
        
        for epoch in range(1, epochs_phaseB + 1):
            model.train()
            pbar = tqdm(dl, desc=f"[PhaseB] Epoch {epoch}/{epochs_phaseB}")
            for batch in pbar:
                # 处理批次数据，支持有无视图的情况
                if len(batch) == 2:
                    u, Fsrc = batch
                    u_view1 = u_view2 = None
                elif len(batch) == 4:
                    u, Fsrc, u_view1, u_view2 = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                
                u = u.to(device)
                Fsrc = Fsrc.to(device)
                u0 = u[:, 0, :]
                u_true = u[:, 1:1+rollout_steps, :]
                
                # 关键实现细节：使用真实轨迹u_true来构造Θ(u)、u_t
                # 计算物理损失时，使用真实数据而不是预测数据
                u_phy_input = u[:, :1+rollout_steps, :]  # 使用真实轨迹
                
                # Phase B中使用中心差分和时间平滑
                # 1. 对时间序列进行平滑
                u_smooth = u_phy_input
                if cfg.get('model', {}).get('use_smoothing', False):
                    u_smooth = time_smoothing(u_phy_input, kernel_size=3)
                
                # 2. 使用中心差分计算时间导数
                u_t_central = dt_central(u_smooth, dt=dt)
                
                # 3. 调整输入序列以匹配中心差分的输出长度
                u_for_lib = u_smooth[:, 1:-1, :]  # 去掉首尾各一个时间步
                
                # 4. 计算物理损失
                if cfg.get('model', {}).get('use_autodiff', False):
                    # 自动微分版本
                    loss_phy, theta_info = model.loss_phy(u_for_lib, dt=dt, dx=dx, F=Fsrc)
                else:
                    # 传统版本：直接使用中心差分计算的u_t
                    # 将真实的u_t_central传递给phys方法以进行正确的标准化
                    u_t_hat, Theta = model.phys(u_for_lib, dx=dx, F=Fsrc, u_t=u_t_central)
                    loss_phy = F.mse_loss(u_t_hat, u_t_central)
                    theta_info = Theta
                
                # 计算L1正则化损失
                l1_loss = l1_penalty()
                
                # 总损失：物理损失 + L1正则化
                loss = loss_phy + lambda_l1 * l1_loss
                
                # 可选的非常小的一致性损失
                if use_con and lambda_con_B > 0:
                    if u_view1 is not None and u_view2 is not None:
                        u_view1 = u_view1.to(device)
                        u_view2 = u_view2.to(device)
                        view1_0 = u_view1[:, 0, :]
                        view2_0 = u_view2[:, 0, :]
                        loss = loss + lambda_con_B * model.loss_con(
                            u0, Fsrc=Fsrc, u_view1=view1_0, u_view2=view2_0
                        )
                    else:
                        loss = loss + lambda_con_B * model.loss_con(u0, Fsrc=Fsrc)

                opt_B.zero_grad()
                loss.backward()
                opt_B.step()

                if use_con:
                    model.ema_update(ema_tau)

                pbar.set_postfix({"phy": float(loss_phy.item()), "l1": float(l1_loss.item()), "lr": opt_B.param_groups[0]['lr']})
            
            # 学习率调度器更新
            scheduler_B.step()

            # 每个 epoch 打印当前发现的方程和训练信息
            if epoch % 5 == 0 or epoch == epochs_phaseB:
                # 获取一个批次的数据来生成方程
                u_sample, _ = next(iter(dl))
                u_sample = u_sample.to(device)
                print("[PhaseB] Discovered:", model.discovered_equation(u_sample, dx, thresh=1e-3))
                
                # 打印训练信息
                log_training_info(u_phy_input, theta_info, None)

    # -------- Phase C: 联合小步微调 (可选) --------
    # 目的：让预测性能与方程结构兼容，并增强跨初值泛化
    # 固定 ξ 的 support（很关键）：即“已经确定非零项集合后不再变动”
    if epochs_phaseC > 0:
        print("===== Starting Phase C: Joint Fine-tuning =====")
        
        # 固定 ξ 的 support（确定非零项集合后不再变动）
        with torch.no_grad():
            xi_proj = model.phys._project_xi()
            xi_support = (torch.abs(xi_proj) > 1e-3).float()  # 确定support
            print(f"[PhaseC] Fixed ξ support: {xi_support.tolist()}")
            
            # 创建一个固定的support掩码
            model.phys.register_buffer('xi_support', xi_support)
        
        # 解冻动态参数
        for param in model.dyn.parameters():
            param.requires_grad = True
        
        # 确保ξ参数可更新，但会被support约束
        if hasattr(model.phys, 'xi'):
            model.phys.xi.requires_grad = True
        
        # 创建联合优化器
        all_params = list(model.dyn.parameters()) + [model.phys.xi]
        if hasattr(model.phys, 'gate_logits'):
            all_params.append(model.phys.gate_logits)
        
        opt_C = torch.optim.Adam(
            all_params,
            lr=float(train_cfg["lr"]) * 0.01,  # 使用非常小的学习率进行微调
            weight_decay=float(train_cfg.get("weight_decay", 0.0))
        )
        
        # 实现一个修改版的_project_xi方法，固定support
        original_project_xi = model.phys._project_xi
        
        def project_xi_with_fixed_support():
            xi_proj = original_project_xi()
            return xi_proj * model.phys.xi_support
        
        model.phys._project_xi = project_xi_with_fixed_support
        
        for epoch in range(1, epochs_phaseC + 1):
            model.train()
            pbar = tqdm(dl, desc=f"[PhaseC] Epoch {epoch}/{epochs_phaseC}")
            for batch in pbar:
                # 处理批次数据，支持有无视图的情况
                if len(batch) == 2:
                    u, Fsrc = batch
                    u_view1 = u_view2 = None
                elif len(batch) == 4:
                    u, Fsrc, u_view1, u_view2 = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                
                u = u.to(device)
                Fsrc = Fsrc.to(device)
                u0 = u[:, 0, :]
                u_true = u[:, 1:1+rollout_steps, :]
                
                # 预测损失
                u_pred = model.rollout(u0, steps=rollout_steps, Fsrc=Fsrc if model.cfg.use_forcing else None)
                loss_rec = model.loss_rec(u_pred, u_true)
                
                # 物理损失（使用真实数据）
                u_phy_input = u[:, :1+rollout_steps, :]
                loss_phy, theta_info = model.loss_phy(u_phy_input, dt=dt, dx=dx)
                
                # 总损失：预测损失 + 物理约束 + 一致性损失
                loss = loss_rec + alpha_C * loss_phy
                
                # 一致性损失
                if use_con and lambda_con_C > 0:
                    if u_view1 is not None and u_view2 is not None:
                        u_view1 = u_view1.to(device)
                        u_view2 = u_view2.to(device)
                        view1_0 = u_view1[:, 0, :]
                        view2_0 = u_view2[:, 0, :]
                        loss = loss + lambda_con_C * model.loss_con(
                            u0, Fsrc=Fsrc, u_view1=view1_0, u_view2=view2_0
                        )
                    else:
                        loss = loss + lambda_con_C * model.loss_con(u0, Fsrc=Fsrc)

                opt_C.zero_grad()
                loss.backward()
                opt_C.step()

                if use_con:
                    model.ema_update(ema_tau)

                pbar.set_postfix({"rec": float(loss_rec.item()), "phy": float(loss_phy.item())})

            # 每个 epoch 打印当前发现的方程和训练信息
            if epoch % 5 == 0 or epoch == epochs_phaseC:
                # 获取一个批次的数据来生成方程
                u_sample, _ = next(iter(dl))
                u_sample = u_sample.to(device)
                print("[PhaseC] Discovered:", model.discovered_equation(u_sample, dx, thresh=1e-3))
                
                # 打印训练信息
                log_training_info(u_phy_input, theta_info, None)
    
    # 保存 checkpoint
    # 获取一个批次的数据来生成最终方程
    u_sample, _ = next(iter(dl))
    u_sample = u_sample.to(device)
    
    payload = {
        "model_state": model.state_dict(),
        "meta": meta,
        "config": cfg,
        "equation": model.discovered_equation(u_sample, dx, thresh=1e-3)
    }
    save_ckpt(ckpt_path, payload)
    print(f"Saved checkpoint to {ckpt_path}")
    print("Final discovered equation:", payload["equation"])
