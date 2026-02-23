from pathlib import Path
import torch

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_ckpt(path: str, payload: dict):
    ensure_dir(str(Path(path).parent))
    torch.save(payload, path)

def load_ckpt(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)
