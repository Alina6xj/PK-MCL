import torch
import torch.nn.functional as F

def ddx_central(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    1D periodic central difference for spatial derivative.
    u: [..., Nx]
    """
    return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0 * dx)

def ddxx_central(u: torch.Tensor, dx: float) -> torch.Tensor:
    """1D periodic second derivative central difference."""
    return (torch.roll(u, shifts=-1, dims=-1) - 2.0 * u + torch.roll(u, shifts=1, dims=-1)) / (dx * dx)

def dt_forward(u_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Forward difference in time for sequence.
    u_seq: [B, T, Nx]
    returns: [B, T-1, Nx]
    """
    return (u_seq[:, 1:, :] - u_seq[:, :-1, :]) / dt

def dt_central(u_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Central difference in time for sequence.
    u_seq: [B, T, Nx]
    returns: [B, T-2, Nx]
    """
    return (u_seq[:, 2:, :] - u_seq[:, :-2, :]) / (2.0 * dt)

def time_smoothing(u_seq: torch.Tensor, kernel_size: int = 3, mode: str = "avg") -> torch.Tensor:
    """
    Light time smoothing.
    u_seq: [B, T, Nx]
    mode: "avg" for average pooling, "gaussian" for gaussian blur
    returns: [B, T, Nx]
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    padding = (kernel_size - 1) // 2
    
    if mode == "avg":
        # 使用1D平均池化进行时间平滑
        u_smoothed = F.avg_pool1d(
            u_seq.transpose(1, 2),  # [B, Nx, T]
            kernel_size=kernel_size,
            padding=padding,
            stride=1
        ).transpose(1, 2)  # [B, T, Nx]
    elif mode == "gaussian":
        # 创建高斯核
        kernel = torch.exp(-torch.linspace(-3, 3, kernel_size) ** 2 / 2) / (torch.sqrt(2 * torch.pi))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, 1).to(u_seq.device)
        
        # 对时间维度应用卷积
        u_smoothed = F.conv2d(u_seq.unsqueeze(1), kernel, padding=(padding, 0), groups=1).squeeze(1)
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")
    
    return u_smoothed

def ddx_spectral(u: torch.Tensor, dx: float, mode: str = "rfft") -> torch.Tensor:
    """
    Spectral derivative using FFT.
    u: [..., Nx]
    mode: "rfft" for real inputs, "fft" for complex inputs
    """
    # 保存原始形状
    original_shape = u.shape
    u = u.contiguous()
    
    # 计算频率域的波数向量
    Nx = u.shape[-1]
    k = torch.fft.fftfreq(Nx, d=dx).to(u.device)  # 获取波数
    k = k * 2 * torch.pi  # 转换为角频率
    
    if mode == "rfft":
        # 实数FFT
        U = torch.fft.rfft(u, dim=-1)
        k = k[:U.shape[-1]]  # 只保留rfft对应的波数
        U_deriv = 1j * k * U  # 频率域导数
        u_deriv = torch.fft.irfft(U_deriv, n=Nx, dim=-1)
    else:
        # 复数FFT
        U = torch.fft.fft(u, dim=-1)
        U_deriv = 1j * k * U  # 频率域导数
        u_deriv = torch.fft.ifft(U_deriv, dim=-1).real
    
    # 恢复原始形状
    u_deriv = u_deriv.view(original_shape)
    
    return u_deriv

def ddxx_spectral(u: torch.Tensor, dx: float, mode: str = "rfft") -> torch.Tensor:
    """
    Spectral second derivative using FFT.
    u: [..., Nx]
    mode: "rfft" for real inputs, "fft" for complex inputs
    """
    # 保存原始形状
    original_shape = u.shape
    u = u.contiguous()
    
    # 计算频率域的波数向量
    Nx = u.shape[-1]
    k = torch.fft.fftfreq(Nx, d=dx).to(u.device)  # 获取波数
    k = k * 2 * torch.pi  # 转换为角频率
    
    if mode == "rfft":
        # 实数FFT
        U = torch.fft.rfft(u, dim=-1)
        k = k[:U.shape[-1]]  # 只保留rfft对应的波数
        U_deriv = -k**2 * U  # 频率域二阶导数
        u_deriv = torch.fft.irfft(U_deriv, n=Nx, dim=-1)
    else:
        # 复数FFT
        U = torch.fft.fft(u, dim=-1)
        U_deriv = -k**2 * U  # 频率域二阶导数
        u_deriv = torch.fft.ifft(U_deriv, dim=-1).real
    
    # 恢复原始形状
    u_deriv = u_deriv.view(original_shape)
    
    return u_deriv
