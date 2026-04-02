from __future__ import annotations

import torch

from typing import Any
from pathlib import Path

def load_metadata(dataset_dir: str | Path, dataset_name: str) -> dict[str, str]:
    """Load UCI metadata text file into a dictionary."""
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / dataset_name / f"{dataset_name}.txt"

    meta: dict[str, str] = {}
    with open(meta_path, "r") as f:
        for line in f:
            key, value = line.split()
            meta[key] = value
    return meta

def is_binary_dataset(dataset_dir: str | Path, dataset_name: str, max_tot: int) -> bool:
    """
    Apply the same filtering logic as the existing code.

    Conditions:
        - metadata file exists
        - binary classification only
        - n_tot <= max_tot
        - no explicit held-out external test split
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / dataset_name / f"{dataset_name}.txt"
    if not meta_path.is_file():
        return False

    meta = load_metadata(dataset_dir, dataset_name)
    num_classes = int(meta["n_clases="])
    n_train_val = int(meta["n_patrons1="])
    n_test = int(meta["n_patrons2="]) if "n_patrons2=" in meta else 0
    n_tot = n_train_val + n_test

    return num_classes == 2 and n_tot <= max_tot and n_test == 0

def list_binary_datasets(config: dict[str, Any]) -> list[str]:
    """List all filtered binary datasets."""
    data_dir = Path(config["data_dir"])
    dataset_names: list[str] = []

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_binary_dataset(config["data_dir"], entry.name, config["max_tot"]):
            dataset_names.append(entry.name)

    return dataset_names

def load_arff_data(
    dataset_dir: str | Path,
    dataset_name: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load ARFF file.

    Returns:
        X: (n, d)
        y: (n,)
    """
    dataset_dir = Path(dataset_dir)
    arff_path = dataset_dir / dataset_name / f"{dataset_name}.arff"

    rows: list[list[float]] = []
    in_data = False

    with open(arff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data:
                rows.append(list(map(float, line.split(","))))

    data = torch.tensor(rows, dtype=dtype)
    X = data[:, :-1]
    y = data[:, -1].long()
    return X, y

def load_validation_split(dataset_dir: str | Path, dataset_name: str) -> tuple[list[int], list[int]]:
    """Load the train/validation split from conxuntos.dat."""
    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / dataset_name / "conxuntos.dat"

    with open(split_path, "r") as f:
        lines = [list(map(int, line.split())) for line in f.readlines()]

    train_idx = lines[0]
    val_idx = lines[1]
    return train_idx, val_idx

def load_kfold_splits(dataset_dir: str | Path, dataset_name: str) -> list[tuple[list[int], list[int]]]:
    """
    Load 4-fold train/test splits from conxuntos_kfold.dat.

    Returns:
        [(train_idx, test_idx), ...]
    """
    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / dataset_name / "conxuntos_kfold.dat"

    with open(split_path, "r") as f:
        lines = [list(map(int, line.split())) for line in f.readlines()]

    folds: list[tuple[list[int], list[int]]] = []
    for repeat in range(4):
        train_idx = lines[2 * repeat]
        test_idx = lines[2 * repeat + 1]
        folds.append((train_idx, test_idx))
    return folds

def subset_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index a dataset using Python index lists."""
    idx = torch.tensor(indices, dtype=torch.long)
    return X[idx], y[idx]