import json
import torch

from pathlib import Path

from nn_jf.models.model import NN
from nn_jf.models.nfm import compute_layerwise_nfm
from nn_jf.models.agop import compute_layerwise_agop
from nn_jf.utils.utils import ensure_dir, load_config, save_json, save_torch, tensor_dtype
from nn_jf.utils.data import list_binary_datasets, load_arff_data, load_kfold_splits, subset_dataset

CONFIG_PATH = "nn.yml"

def load_best_params(path: str | Path) -> dict:
    """Load the best hyperparameters found by train.py."""

    with open(path, "r") as f:
        return json.load(f)

def save_layerwise_matrices(path: str | Path, matrices: list[torch.Tensor]) -> None:
    """Save a list of layerwise matrices as a keyed torch payload."""
    payload = {f"layer_{idx}": matrix.detach().cpu() for idx, matrix in enumerate(matrices)}
    save_torch(path, payload)

def average_layerwise_matrices(layerwise_matrices: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Average layerwise matrices across folds."""
    if len(layerwise_matrices) == 0:
        raise ValueError("layerwise_matrices must be non-empty.")

    num_layers = len(layerwise_matrices[0])
    averaged: list[torch.Tensor] = []

    for layer_idx in range(num_layers):
        stacked = torch.stack(
            [fold_matrices[layer_idx].detach().cpu() for fold_matrices in layerwise_matrices],
            dim=0,
        )
        averaged.append(stacked.mean(dim=0))

    return averaged

def evaluate_one_dataset(config: dict, dataset_name: str) -> None:
    """
    Run 4-fold evaluation using the best hyperparameters from train.py.

    Save only:
        - dataset-level average test accuracy
        - dataset-level average layerwise AGOP
        - dataset-level average layerwise NFM
    """
    dtype = tensor_dtype(config)
    dataset_result_dir = ensure_dir(Path(config["save_dir"]) / dataset_name)
    best_params = load_best_params(dataset_result_dir / "best_params.json")

    X, y = load_arff_data(
        dataset_dir=config["data_dir"],
        dataset_name=dataset_name,
        dtype=dtype,
    )
    folds = load_kfold_splits(config["data_dir"], dataset_name)

    test_accuracies: list[float] = []
    all_fold_agops: list[list[torch.Tensor]] = []
    all_fold_nfms: list[list[torch.Tensor]] = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train = subset_dataset(X, y, train_idx)
        X_test, y_test = subset_dataset(X, y, test_idx)

        model = NN(
            config=config,
            input_dim=X.shape[1],
            num_layers=int(best_params["num_layers"]),
            h_dim=int(best_params["h_dim"]),
            use_batch_norm=bool(best_params["use_batch_norm"]),
            lr=float(best_params["lr"]),
        )
        model.fit(X_train=X_train, y_train=y_train)

        test_acc = model.score(X_test, y_test)
        agops = compute_layerwise_agop(model, X_train.to(model.device, dtype=model.dtype))
        nfms = compute_layerwise_nfm(model)

        test_accuracies.append(float(test_acc))
        all_fold_agops.append(agops)
        all_fold_nfms.append(nfms)

        print(f"[eval] dataset={dataset_name} fold={fold_id} test_acc={test_acc:.6f}")

    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    avg_agops = average_layerwise_matrices(all_fold_agops)
    avg_nfms = average_layerwise_matrices(all_fold_nfms)

    save_json(
        dataset_result_dir / "eval_summary.json",
        {
            "dataset_name": dataset_name,
            "avg_test_accuracy": avg_test_acc,
        },
    )
    save_layerwise_matrices(dataset_result_dir / "avg_layerwise_agop.pt", avg_agops)
    save_layerwise_matrices(dataset_result_dir / "avg_layerwise_nfm.pt", avg_nfms)

    print(f"[eval] dataset={dataset_name} avg_test_acc={avg_test_acc:.6f}")

def main() -> None:
    """Evaluate all filtered binary datasets with the best hyperparameters from train.py."""
    config = load_config(CONFIG_PATH)
    dataset_names = list_binary_datasets(config)

    print(f"[eval] found {len(dataset_names)} filtered binary datasets")

    for dataset_name in dataset_names:
        best_params_path = Path(config["save_dir"]) / dataset_name / "best_params.json"
        if not best_params_path.is_file():
            print(f"[eval] skipping {dataset_name}: missing train.py output")
            continue
        evaluate_one_dataset(config, dataset_name)

if __name__ == "__main__":
    main()