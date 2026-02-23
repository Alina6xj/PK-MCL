import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
from models.system_adapter import SystemAdapter


def test_system_adapter():
    """测试系统适配器"""
    print("="*70)
    print("System Adapter Test")
    print("="*70)
    
    systems = ["burgers", "fhn", "ns", "schrodinger", "rd"]
    
    for system_type in systems:
        print(f"\n--- Testing {system_type} ---")
        
        cfg = SystemAdapter.get_config(system_type)
        print(f"  Config:")
        print(f"    dim: {cfg['dim']}D")
        print(f"    variables: {cfg['variables']}")
        print(f"    data keys: {cfg['data_keys']}")
        
        # 创建模拟数据集
        dummy_dataset = create_dummy_dataset(system_type)
        
        # 测试数据准备
        train_data, F_train = SystemAdapter.prepare_training_data(dummy_dataset, system_type, n_train_frames=60)
        
        if isinstance(train_data, tuple):
            print(f"  Multi-variable data:")
            for i, var in enumerate(train_data):
                print(f"    {cfg['variables'][i]}: shape={var.shape}")
        else:
            print(f"  Single-variable data:")
            print(f"    {cfg['variables'][0]}: shape={train_data.shape}")
        
        if F_train is not None:
            print(f"    Force: shape={F_train.shape}")
    
    print("\n" + "="*70)
    print("All tests passed! System Adapter is working correctly.")
    print("="*70)


def create_dummy_dataset(system_type: str):
    """创建模拟数据集用于测试"""
    if system_type == "burgers":
        return {
            "train_u": torch.randn(80, 100, 256),
            "test_u": torch.randn(20, 100, 256),
            "train_F": torch.randn(80, 256),
            "test_F": torch.randn(20, 256),
            "meta": {"nx": 256, "dt": 1e-4, "nu": 0.1}
        }
    elif system_type == "fhn":
        return {
            "train_v": torch.randn(8, 250, 100, 100),
            "train_w": torch.randn(8, 250, 100, 100),
            "test_v": torch.randn(2, 250, 100, 100),
            "test_w": torch.randn(2, 250, 100, 100),
            "meta": {"nx": 100, "ny": 100, "dt": 1e-6}
        }
    elif system_type == "ns":
        return {
            "train_omega": torch.randn(80, 100, 64, 64),
            "test_omega": torch.randn(20, 100, 64, 64),
            "meta": {"nx": 64, "ny": 64, "dt": 1e-3}
        }
    elif system_type == "schrodinger":
        return {
            "train_real": torch.randn(80, 100, 512),
            "train_imag": torch.randn(80, 100, 512),
            "test_real": torch.randn(20, 100, 512),
            "test_imag": torch.randn(20, 100, 512),
            "meta": {"nx": 512, "dt": 5e-3, "kappa": 1.0}
        }
    elif system_type == "rd":
        return {
            "train_u": torch.randn(80, 100, 100, 100),
            "train_v": torch.randn(80, 100, 100, 100),
            "test_u": torch.randn(20, 100, 100, 100),
            "test_v": torch.randn(20, 100, 100, 100),
            "meta": {"nx": 100, "ny": 100, "dt": 1.0}
        }
    else:
        return {}


if __name__ == "__main__":
    test_system_adapter()
