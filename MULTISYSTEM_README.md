# PKMCL多系统项目 - 完整实现总结

## 项目概述

本项目已成功扩展PKMCL（Physics-Koopman Multi-scale Contrastive Learning）框架，支持五种不同的物理系统的数据生成和基础实验适配。

## 已支持的系统

### 1. Burgers系统 (1D单变量)
- **方程**：u_t = -u u_x + ν u_xx + F(x)
- **数据生成**：`data/generate_burgers.py`
- **配置文件**：`configs/burgers.yaml`
- **特点**：1D、周期边界、强迫项、多尺度分解

### 2. FHN系统 (2D双变量)
- **方程**：
  - ∂_t v = D_v ∇²v + f(v, w) + I
  - ∂_t w = D_w ∇²w + ε g(v, w)
- **数据生成**：`data/generate_fhn.py`
- **配置文件**：`configs/fhn.yaml`
- **特点**：2D、Neumann边界、双变量（v, w）、高斯初始条件

### 3. NS系统 (2D单变量)
- **方程**：i ∂_t ω + u·∇ω = ν Δω + f, ∇·u=0, ω=∇×u
- **数据生成**：`data/generate_ns.py`
- **配置文件**：`configs/ns.yaml`
- **特点**：2D、周期边界、涡度形式、伪谱方法、RK3时间积分

### 4. 薛定谔方程 (1D复值)
- **方程**：i ∂_t ψ = -∇² ψ - κ |ψ|² ψ
- **数据生成**：`data/generate_schrodinger.py`
- **配置文件**：`configs/schrodinger.yaml`
- **特点**：1D、复值场、分裂步傅里叶方法、波包初始条件

### 5. RD系统 (2D双变量)
- **方程**：
  - ∂_t u = D_u ∇²u - u v² + F (1 - u)
  - ∂_t v = D_v ∇²v + u v² - (F + k) v
- **数据生成**：`data/generate_rd.py`
- **配置文件**：`configs/rd.yaml`
- **特点**：2D、周期边界、双变量（u, v）、反应扩散、方形扰动初始条件

## 文件结构

### 新增文件

```
data/
├── generate_fhn.py          # FHN系统数据生成
├── generate_ns.py           # NS系统数据生成
├── generate_schrodinger.py  # 薛定谔方程数据生成
└── generate_rd.py           # RD系统数据生成

configs/
├── fhn.yaml                 # FHN系统配置
├── ns.yaml                  # NS系统配置
├── schrodinger.yaml         # 薛定谔方程配置
└── rd.yaml                  # RD系统配置

models/
├── koopman_operator_2d.py   # 2D版本的Koopman算子
└── system_adapter.py        # 系统适配器

train/
└── train_multisystem.py     # 多系统训练框架
```

### 修改文件
- `main.py` - 添加了多系统支持，通过配置文件的`system`字段选择系统

## 使用方法

### 生成数据

#### 方法1：通过main.py（推荐）
```bash
# Burgers系统
python main.py --mode train --config configs/burgers.yaml

# FHN系统
python main.py --mode train --config configs/fhn.yaml

# NS系统
python main.py --mode train --config configs/ns.yaml

# 薛定谔方程
python main.py --mode train --config configs/schrodinger.yaml

# RD系统
python main.py --mode train --config configs/rd.yaml
```

#### 方法2：直接调用数据生成脚本
```bash
# FHN
python data/generate_fhn.py --n_traj 100 --n_train 80

# NS
python data/generate_ns.py --n_traj 100 --n_train 80

# 薛定谔
python data/generate_schrodinger.py --n_traj 100 --n_train 80

# RD
python data/generate_rd.py --n_traj 100 --n_train 80
```

## 已完成的工作

### 1. 数据生成
- ✅ 五种系统的完整数据生成实现
- ✅ 数值方法：有限差分、伪谱、分裂步傅里叶、RK2/RK3积分
- ✅ 边界条件：周期、Neumann
- ✅ 初始条件：随机傅里叶、高斯涡旋、波包、方形扰动

### 2. 基础架构
- ✅ 配置文件系统，支持system字段选择
- ✅ 系统适配器（`SystemAdapter`）统一数据格式
- ✅ 2D版本的ConvEncoder/ConvDecoder
- ✅ 2D版本的KoopmanRollout
- ✅ 多系统训练框架占位实现

### 3. 测试验证
- ✅ 所有系统的数据生成测试通过
- ✅ 无NaN/Inf数值不稳定问题
- ✅ 数据集形状和统计特性验证

## 后续工作建议

### 优先级1：完整训练支持（必需）

#### 1.1 扩展PhysicsLibrary
- [ ] 支持2D空间导数（u_x, u_y, u_xx, u_yy, Δu）
- [ ] 支持多变量系统（为每个变量独立构造库）
- [ ] 支持复值运算（薛定谔方程）
- [ ] 添加系统特定的PDE候选项

#### 1.2 集成2D Koopman模型
- [ ] 修改`pkmcl.py`，根据系统维度选择1D或2D模型
- [ ] 扩展多尺度分解为2D FFT
- [ ] 支持多变量输入（分别处理每个变量）

#### 1.3 修改训练循环
- [ ] 适配多系统数据格式
- [ ] 支持多变量损失计算
- [ ] 处理复值数据
- [ ] 为每个系统配置合适的物理库

### 优先级2：高级功能（推荐）

#### 2.1 系统特定优化
- [ ] 为薛定谔方程添加复值MLP
- [ ] 为2D系统优化谱导数计算
- [ ] 多变量一致性损失

#### 2.2 评估和可视化
- [ ] 多系统评估脚本
- [ ] 2D场可视化工具
- [ ] 物理方程发现质量评估

### 优先级3：长期优化（可选）

#### 3.1 性能优化
- [ ] 大规模数据集预生成
- [ ] 混合精度训练
- [ ] 数据加载优化

#### 3.2 更多系统
- [ ] KdV方程
- [ ] Kuramoto-Sivashinsky方程
- [ ] 其他PDE系统

## 技术细节

### 数值方法总结

| 系统 | 空间离散 | 时间积分 | 边界条件 |
|------|----------|----------|----------|
| Burgers | 中心差分 | RK2 | 周期 |
| FHN | 中心差分 | RK2 | Neumann |
| NS | 伪谱（FFT） | RK3 | 周期 |
| Schrodinger | 伪谱（FFT） | 分裂步 | 周期 |
| RD | 中心差分 | 显式 | 周期 |

### 数据集统计

| 系统 | 网格 | 时间步长 | 帧数 | 轨迹数 |
|------|------|----------|------|--------|
| Burgers | 256×1 | 1e-4 | 100 | 120 |
| FHN | 100×100 | 1e-6 | 250 | 10 |
| NS | 64×64 | 1e-3 | 100 | 100 |
| Schrodinger | 512×1 | 5e-3 | 100 | 100 |
| RD | 100×100 | 1.0 | 100 | 100 |

## 贡献指南

要添加新系统，请遵循以下步骤：

1. 创建数据生成脚本 `data/generate_newsystem.py`
2. 创建配置文件 `configs/newsystem.yaml`
3. 在 `main.py` 中添加系统类型判断
4. （可选）扩展PhysicsLibrary和模型架构
5. 添加测试脚本验证数据生成

## 引用

如果使用本项目，请考虑引用相关文献：
- PKML方法原始论文
- 各物理系统的参考文献
- （待补充）

## 许可证

（与原项目保持一致）
