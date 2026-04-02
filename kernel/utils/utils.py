import yaml
import json
import torch
import random
import numpy as np

from typing import Any
from pathlib import Path

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}

def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    """Set Python / NumPy / Torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON with stable formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def tensor_dtype(config: dict[str, Any]) -> torch.dtype:
    """Resolve tensor dtype from config."""
    return DTYPE_MAP[config["dtype"]]

def tensor_device(config: dict[str, Any]) -> torch.device:
    """Resolve device from config, falling back to CPU if CUDA is unavailable."""
    if config["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Binary classification accuracy."""
    return float((y_true == y_pred).float().mean().item())