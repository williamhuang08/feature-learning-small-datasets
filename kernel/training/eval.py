from pathlib import Path

from kernel.models.model import KernelSVM
from kernel.utils.utils import load_config, save_json, tensor_dtype
from kernel.utils.data import list_binary_datasets, load_arff_data, load_kfold_splits, subset_dataset

CONFIG_PATH = "kernel.yml"

def result_root(config: dict) -> Path:
    """Return the kernel-specific result root."""
    return Path(config["save_dir"]) / config["kernel"]

def load_best_params(path: str | Path) -> dict:
    """Load the best hyperparameters found by train.py."""
    import json

    with open(path, "r") as f:
        return json.load(f)

# kernel/eval.py
def extract_kernel_params(best_params: dict) -> dict[str, float | int]:
    """Extract kernel-specific parameters from saved best_params."""
    kernel_name = best_params["kernel"]

    if kernel_name == "gaussian":
        return {"gamma": float(best_params["gamma"])}

    if kernel_name == "laplace":
        return {"gamma": float(best_params["gamma"])}

    if kernel_name == "polynomial":
        return {
            "degree": int(best_params["degree"]),
            "gamma": float(best_params["gamma"]),
            "coef0": float(best_params["coef0"]),
        }

    if kernel_name == "ntk":
        return {
            "num_layers": int(best_params["num_layers"]),
            "num_fixed_layers": int(best_params["num_fixed_layers"]),
        }

    raise ValueError(f"Unsupported kernel: {kernel_name}")

def evaluate_one_dataset(config: dict, dataset_name: str) -> None:
    """Run 4-fold evaluation and save only the dataset-level average test accuracy."""
    dtype = tensor_dtype(config)
    dataset_result_dir = result_root(config) / dataset_name
    best_params = load_best_params(dataset_result_dir / "best_params.json")

    X, y = load_arff_data(
        dataset_dir=config["data_dir"],
        dataset_name=dataset_name,
        dtype=dtype,
    )
    folds = load_kfold_splits(config["data_dir"], dataset_name)

    test_accuracies: list[float] = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train = subset_dataset(X, y, train_idx)
        X_test, y_test = subset_dataset(X, y, test_idx)

        model = KernelSVM(
            config=config,
            input_dim=X.shape[1],
            kernel_name=str(best_params["kernel"]),
            kernel_params=extract_kernel_params(best_params),
            c_value=float(best_params["c_value"]),
        )
        model.fit(X_train=X_train, y_train=y_train)

        test_acc = model.score(
            X_test.to(model.device, dtype=model.dtype),
            y_test.to(model.device),
        )
        test_accuracies.append(float(test_acc))

        print(
            f"[eval] kernel={config['kernel']} "
            f"dataset={dataset_name} "
            f"fold={fold_id} "
            f"test_acc={test_acc:.6f}"
        )

    avg_test_acc = sum(test_accuracies) / len(test_accuracies)

    save_json(
        dataset_result_dir / "eval_summary.json",
        {
            "dataset_name": dataset_name,
            "kernel": config["kernel"],
            "avg_test_accuracy": avg_test_acc,
        },
    )

    print(
        f"[eval] kernel={config['kernel']} "
        f"dataset={dataset_name} "
        f"avg_test_acc={avg_test_acc:.6f}"
    )

def main() -> None:
    """Evaluate all filtered binary datasets with the best hyperparameters from train.py."""
    config = load_config(CONFIG_PATH)
    dataset_names = list_binary_datasets(config)

    print(f"[eval] kernel={config['kernel']} found {len(dataset_names)} filtered binary datasets")

    for dataset_name in dataset_names:
        best_params_path = result_root(config) / dataset_name / "best_params.json"
        if not best_params_path.is_file():
            print(f"[eval] skipping {dataset_name}: missing train.py output")
            continue
        evaluate_one_dataset(config, dataset_name)

if __name__ == "__main__":
    main()