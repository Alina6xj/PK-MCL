import torch


def spectral_derivative(u: torch.Tensor, dx: float, order: int = 1) -> torch.Tensor:
    """
    使用谱方法计算空间导数。
    
    Args:
        u: [..., Nx] - 输入信号，最后一维是空间维度
        dx: float - 空间步长
        order: int - 导数阶数（1或2）
        
    Returns:
        du/dx: [..., Nx] - 空间导数
    """
    # 获取输入形状
    shape = u.shape
    Nx = shape[-1]
    
    # 计算波数向量 k
    # 对于实信号，只需要正频率部分
    k = torch.fft.fftfreq(Nx, d=dx).to(u.device) * 2 * torch.pi
    k = torch.fft.fftshift(k)
    
    # 计算FFT
    u_fft = torch.fft.fft(u, dim=-1)
    u_fft = torch.fft.fftshift(u_fft, dim=-1)
    
    # 应用导数算子
    if order == 1:
        # 一阶导数：i*k
        derivative_operator = 1j * k
    elif order == 2:
        # 二阶导数：-k^2
        derivative_operator = -(k ** 2)
    else:
        raise ValueError(f"Unsupported derivative order: {order}. Only 1 or 2 are supported.")
    
    # 计算导数的FFT
    du_fft = u_fft * derivative_operator.view(*([1] * (len(shape) - 1)), -1)
    
    # 逆FFT回到时域
    du_fft = torch.fft.ifftshift(du_fft, dim=-1)
    du = torch.fft.ifft(du_fft, dim=-1).real
    
    return du


def spectral_ddx(u: torch.Tensor, dx: float) -> torch.Tensor:
    """1D periodic spectral first derivative."""
    return spectral_derivative(u, dx, order=1)


def spectral_ddxx(u: torch.Tensor, dx: float) -> torch.Tensor:
    """1D periodic spectral second derivative."""
    return spectral_derivative(u, dx, order=2)


def spectral_derivative_multiscale(u_low: torch.Tensor, u_mid: torch.Tensor, u_high: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对多尺度分解后的信号计算谱导数。
    
    Args:
        u_low: [..., Nx] - 低频分量
        u_mid: [..., Nx] - 中频分量
        u_high: [..., Nx] - 高频分量
        dx: float - 空间步长
        
    Returns:
        (u_low_x, u_low_xx, u_mid_x, u_mid_xx, u_high_x, u_high_xx) - 各分量的一阶和二阶导数
    """
    # 计算各分量的一阶导数
    u_low_x = spectral_ddx(u_low, dx)
    u_mid_x = spectral_ddx(u_mid, dx)
    u_high_x = spectral_ddx(u_high, dx)
    
    # 计算各分量的二阶导数
    u_low_xx = spectral_ddxx(u_low, dx)
    u_mid_xx = spectral_ddxx(u_mid, dx)
    u_high_xx = spectral_ddxx(u_high, dx)
    
    return u_low_x, u_low_xx, u_mid_x, u_mid_xx, u_high_x, u_high_xx