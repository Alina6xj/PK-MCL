# PK-MCL: Physics-Koopman Multi-scale Contrastive Learning

## 项目概述

本项目提供一个可直接运行的 PyTorch 工程，实现了PK-MCL方法，输出：
1. 可滚动预测的动力学演化算子（Koopman 潜空间线性演化 + 编解码器）
2. 显式可解释控制方程（候选库 Θ(u) + 稀疏系数 ξ）

### 已支持的物理系统

| 系统 | 维度 | 变量数 | 说明 |
|------|------|--------|------|
| **Burgers** | 1D | 1 | 含强迫项的粘性Burgers方程 |
| **FHN** | 2D | 2 | FitzHugh-Nagumo反应扩散系统 |
| **NS** | 2D | 1 | 二维不可压缩Navier-Stokes方程（涡度形式） |
| **Schrodinger** | 1D | 2 (复) | 非线性薛定谔方程 |
| **RD** | 2D | 2 | Gray-Scott反应扩散系统 |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用不同系统

#### 1. Burgers系统 (1D)
```bash
# 生成数据（可选，会自动生成）
python -m data.generate_burgers --out data/burgers_dataset.pt

# 训练
python main.py --mode train --config configs/burgers.yaml

# 测试
python main.py --mode test --config configs/burgers.yaml --ckpt checkpoints/pkmcl_burgers.pt
```

#### 2. FHN系统 (2D双变量)
```bash
# 生成数据（可选，会自动生成）
python data/generate_fhn.py --out data/fhn_dataset.pt

# 训练
python main.py --mode train --config configs/fhn.yaml
```

#### 3. NS系统 (2D)
```bash
# 生成数据（可选，会自动生成）
python data/generate_ns.py --out data/ns_dataset.pt

# 训练
python main.py --mode train --config configs/ns.yaml
```

#### 4. 薛定谔方程 (1D复值)
```bash
# 生成数据（可选，会自动生成）
python data/generate_schrodinger.py --out data/schrodinger_dataset.pt

# 训练
python main.py --mode train --config configs/schrodinger.yaml
```

#### 5. RD系统 (2D双变量)
```bash
# 生成数据（可选，会自动生成）
python data/generate_rd.py --out data/rd_dataset.pt

# 训练
python main.py --mode train --config configs/rd.yaml
```

## 项目结构

```
pkmcl_burgers/
├── data/                          # 数据生成脚本
│   ├── generate_burgers.py       # Burgers系统
│   ├── generate_fhn.py           # FHN系统
│   ├── generate_ns.py            # NS系统
│   ├── generate_schrodinger.py   # 薛定谔方程
│   └── generate_rd.py            # RD系统
├── configs/                       # 配置文件
│   ├── burgers.yaml              # Burgers配置
│   ├── fhn.yaml                  # FHN配置
│   ├── ns.yaml                   # NS配置
│   ├── schrodinger.yaml          # 薛定谔配置
│   └── rd.yaml                   # RD配置
├── models/                        # 模型组件
│   ├── koopman_operator.py       # 1D Koopman算子
│   ├── koopman_operator_2d.py    # 2D Koopman算子
│   ├── physics_library.py        # 物理库
│   └── system_adapter.py         # 系统适配器
├── train/                         # 训练模块
│   ├── train.py                  # 原训练脚本
│   └── train_multisystem.py      # 多系统训练框架
├── utils/                         # 工具函数
├── main.py                        # 主程序入口
└── MULTISYSTEM_README.md         # 详细的多系统文档
```

## 各系统详细说明

### Burgers系统
- **方程**: u_t = -u u_x + ν u_xx + F(x)
- **方法**: 中心差分 + RK2积分
- **边界**: 周期边界条件
- **初始条件**: 随机傅里叶叠加

### FHN系统
- **方程**: FitzHugh-Nagumo反应扩散
- **方法**: 中心差分 + RK2积分
- **边界**: Neumann边界条件
- **初始条件**: 高斯扰动

### NS系统
- **方程**: 二维Navier-Stokes（涡度形式）
- **方法**: 伪谱方法 + RK3积分
- **边界**: 周期边界条件
- **初始条件**: 高斯涡旋

### 薛定谔方程
- **方程**: i ∂_t ψ = -∇² ψ - κ |ψ|² ψ
- **方法**: 分裂步傅里叶方法
- **边界**: 周期边界条件
- **初始条件**: 随机波包

### RD系统
- **方程**: Gray-Scott反应扩散
- **方法**: 显式有限差分
- **边界**: 周期边界条件
- **初始条件**: 方形随机扰动

## 后续工作

要完成所有系统的完整训练支持，还需要：

1. 扩展PhysicsLibrary支持2D空间导数和多变量系统
2. 集成KoopmanRollout2D到主模型架构
3. 处理复值数据（薛定谔方程）
4. 适配训练循环到不同数据格式

详细的技术路线图请参考 `MULTISYSTEM_README.md`。

## 参考文献

- PK-MCL原始论文
- 各物理系统的相关文献
