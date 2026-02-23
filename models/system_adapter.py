import torch
from typing import Dict, Any, Tuple, Optional


class SystemAdapter:
    """
    系统适配器，用于统一处理不同系统的数据集格式
    """
    
    @staticmethod
    def get_config(system_type: str) -> Dict[str, Any]:
        """获取系统配置"""
        configs = {
            "burgers": {
                "dim": 1,
                "variables": ["u"],
                "data_keys": {"train": "train_u", "test": "test_u", "force": "train_F"},
                "shape": ["T", "Nx"],
            },
            "fhn": {
                "dim": 2,
                "variables": ["v", "w"],
                "data_keys": {"train_v": "train_v", "train_w": "train_w", "test_v": "test_v", "test_w": "test_w"},
                "shape": ["T", "Ny", "Nx"],
            },
            "ns": {
                "dim": 2,
                "variables": ["omega"],
                "data_keys": {"train": "train_omega", "test": "test_omega"},
                "shape": ["T", "Ny", "Nx"],
            },
            "schrodinger": {
                "dim": 1,
                "variables": ["real", "imag"],
                "data_keys": {"train_real": "train_real", "train_imag": "train_imag", "test_real": "test_real", "test_imag": "test_imag"},
                "shape": ["T", "Nx"],
            },
            "rd": {
                "dim": 2,
                "variables": ["u", "v"],
                "data_keys": {"train_u": "train_u", "train_v": "train_v", "test_u": "test_u", "test_v": "test_v"},
                "shape": ["T", "Ny", "Nx"],
            },
        }
        return configs.get(system_type, configs["burgers"])
    
    @staticmethod
    def prepare_training_data(dataset: Dict[str, Any], system_type: str, n_train_frames: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        准备训练数据，统一格式
        
        Returns:
            u_train: [B, T, ...] - 训练数据
            F_train: [B, ...] or None - 强迫项（如果有）
        """
        cfg = SystemAdapter.get_config(system_type)
        
        if system_type == "burgers":
            u_train = dataset["train_u"]
            F_train = dataset.get("train_F", None)
            if n_train_frames is not None:
                u_train = u_train[:, :n_train_frames]
            return u_train, F_train
        
        elif system_type == "fhn":
            v_train = dataset["train_v"]
            w_train = dataset["train_w"]
            if n_train_frames is not None:
                v_train = v_train[:, :n_train_frames]
                w_train = w_train[:, :n_train_frames]
            return (v_train, w_train), None
        
        elif system_type == "ns":
            omega_train = dataset["train_omega"]
            if n_train_frames is not None:
                omega_train = omega_train[:, :n_train_frames]
            return omega_train, None
        
        elif system_type == "schrodinger":
            real_train = dataset["train_real"]
            imag_train = dataset["train_imag"]
            if n_train_frames is not None:
                real_train = real_train[:, :n_train_frames]
                imag_train = imag_train[:, :n_train_frames]
            return (real_train, imag_train), None
        
        elif system_type == "rd":
            u_train = dataset["train_u"]
            v_train = dataset["train_v"]
            if n_train_frames is not None:
                u_train = u_train[:, :n_train_frames]
                v_train = v_train[:, :n_train_frames]
            return (u_train, v_train), None
        
        else:
            return dataset["train_u"], dataset.get("train_F", None)
    
    @staticmethod
    def prepare_test_data(dataset: Dict[str, Any], system_type: str, start_frame: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        准备测试数据，统一格式
        """
        cfg = SystemAdapter.get_config(system_type)
        
        if system_type == "burgers":
            u_test = dataset["test_u"]
            F_test = dataset.get("test_F", None)
            if start_frame is not None:
                u_test = u_test[:, start_frame:]
            return u_test, F_test
        
        elif system_type == "fhn":
            v_test = dataset["test_v"]
            w_test = dataset["test_w"]
            if start_frame is not None:
                v_test = v_test[:, start_frame:]
                w_test = w_test[:, start_frame:]
            return (v_test, w_test), None
        
        elif system_type == "ns":
            omega_test = dataset["test_omega"]
            if start_frame is not None:
                omega_test = omega_test[:, start_frame:]
            return omega_test, None
        
        elif system_type == "schrodinger":
            real_test = dataset["test_real"]
            imag_test = dataset["test_imag"]
            if start_frame is not None:
                real_test = real_test[:, start_frame:]
                imag_test = imag_test[:, start_frame:]
            return (real_test, imag_test), None
        
        elif system_type == "rd":
            u_test = dataset["test_u"]
            v_test = dataset["test_v"]
            if start_frame is not None:
                u_test = u_test[:, start_frame:]
                v_test = v_test[:, start_frame:]
            return (u_test, v_test), None
        
        else:
            return dataset["test_u"], dataset.get("test_F", None)
