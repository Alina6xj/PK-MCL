import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.spectral_diff import spectral_ddx, spectral_ddxx
from utils.diff import ddx_central, ddxx_central

# 创建一个简单的测试函数
x_np = np.linspace(0.0, 2*np.pi, 128, endpoint=False)
dx = x_np[1] - x_np[0]
x = torch.from_numpy(x_np)
u = torch.sin(x) + 0.5 * torch.sin(2*x) + 0.2 * torch.sin(4*x)

# 计算解析导数
u_x_analytical = torch.cos(x) + torch.cos(2*x) + 0.8 * torch.cos(4*x)
u_xx_analytical = -torch.sin(x) - 2 * torch.sin(2*x) - 3.2 * torch.sin(4*x)

# 使用谱导数计算
u_x_spectral = spectral_ddx(u.unsqueeze(0).unsqueeze(0), dx).squeeze()
u_xx_spectral = spectral_ddxx(u.unsqueeze(0).unsqueeze(0), dx).squeeze()

# 使用中心差分计算
u_x_central = ddx_central(u.unsqueeze(0).unsqueeze(0), dx).squeeze()
u_xx_central = ddxx_central(u.unsqueeze(0).unsqueeze(0), dx).squeeze()

# 计算误差
error_x_spectral = torch.mean((u_x_spectral - u_x_analytical)**2)
error_xx_spectral = torch.mean((u_xx_spectral - u_xx_analytical)**2)
error_x_central = torch.mean((u_x_central - u_x_analytical)**2)
error_xx_central = torch.mean((u_xx_central - u_xx_analytical)**2)

print(f"一阶导数误差 (谱导数): {error_x_spectral:.6e}")
print(f"一阶导数误差 (中心差分): {error_x_central:.6e}")
print(f"二阶导数误差 (谱导数): {error_xx_spectral:.6e}")
print(f"二阶导数误差 (中心差分): {error_xx_central:.6e}")

# 绘制结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x.numpy(), u.numpy(), label='Original')
plt.title('Original Function')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x.numpy(), u_x_analytical.numpy(), label='Analytical')
plt.plot(x.numpy(), u_x_spectral.numpy(), label='Spectral', linestyle='--')
plt.plot(x.numpy(), u_x_central.numpy(), label='Central', linestyle=':')
plt.title('First Derivative')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x.numpy(), u_xx_analytical.numpy(), label='Analytical')
plt.plot(x.numpy(), u_xx_spectral.numpy(), label='Spectral', linestyle='--')
plt.plot(x.numpy(), u_xx_central.numpy(), label='Central', linestyle=':')
plt.title('Second Derivative')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x.numpy(), (u_x_spectral - u_x_analytical).numpy(), label='Spectral Error')
plt.plot(x.numpy(), (u_x_central - u_x_analytical).numpy(), label='Central Error')
plt.title('First Derivative Error')
plt.legend()

plt.tight_layout()
plt.savefig('derivative_test.png')
print("Plot saved to derivative_test.png")
