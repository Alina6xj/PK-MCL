from models.physics_library import PhysicsRegressor, PhysicsLibrary, LibraryConfig
import torch

# 创建测试配置
lib_cfg = LibraryConfig(
    include_u=True,
    include_ux=True,
    include_uxx=True,
    include_u2=True,
    include_uux=True
)

# 初始化库和物理回归器
library = PhysicsLibrary(lib_cfg)
phys = PhysicsRegressor(library, projector_type='hard-threshold')

# 创建测试输入
u = torch.randn(2, 3, 64)  # [B, T, Nx]
dx = 0.1

# 测试前向传播
u_t_hat, Theta = phys(u, dx)
print(f"Forward pass successful: u_t_hat shape = {u_t_hat.shape}, Theta shape = {Theta.shape}")

# 测试pretty_equation
print("\nDiscovered equation:")
print(phys.pretty_equation())

# 测试L1正则化
l1_loss = phys.get_l1_regularization(lambda_l1=1e-2)
print(f"\nL1 regularization loss: {l1_loss.item()}")

print("\nAll tests passed!")
