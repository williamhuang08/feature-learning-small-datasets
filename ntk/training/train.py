from pathlib import Path

from ntk.models.model import grid_search_ntk
from ntk.utils.utils import ensure_dir, load_config, save_json, tensor_dtype
from ntk.utils.data import list_binary_datasets, load_arff_data, load_metadata, load_validation_split, subset_dataset

CONFIG_PATH = "ntk.yml"

def train_one_dataset(config: dict, dataset_name: str) -> None:
    """Search best hyperparameters on the train/validation split and save only the result."""
    dataset_dir = ensure_dir(Path(config["save_dir"]) / dataset_name)
    best_params_path = dataset_dir / "best_params.json"

    meta = load_metadata(config["data_dir"], dataset_name)
    num_classes = int(meta["n_clases="])
    input_dim = int(meta["n_entradas="])
    n_train = int(meta["n_patrons_entrena="])
    n_val = int(meta["n_patrons_valida="])
    n_train_val = int(meta["n_patrons1="])
    n_test = int(meta["n_patrons2="]) if "n_patrons2=" in meta else 0
    n_tot = n_train_val + n_test

    print(
        f"[train] dataset={dataset_name} "
        f"n_tot={n_tot} "
        f"d={input_dim} "
        f"c={num_classes} "
        f"n_train={n_train} "
        f"n_val={n_val} "
        f"n_test={n_test}"
    )

    if best_params_path.exists():
        print(f"[train] skipping {dataset_name}: best_params already found")
        return

    dtype = tensor_dtype(config)

    X, y = load_arff_data(
        dataset_dir=config["data_dir"],
        dataset_name=dataset_name,
        dtype=dtype,
    )
    train_idx, val_idx = load_validation_split(config["data_dir"], dataset_name)
    X_train, y_train = subset_dataset(X, y, train_idx)
    X_val, y_val = subset_dataset(X, y, val_idx)

    best_result = grid_search_ntk(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    save_json(
        best_params_path,
        {
            "dataset_name": dataset_name,
            "num_layers": best_result.num_layers,
            "num_fixed_layers": best_result.num_fixed_layers,
            "c_value": best_result.c_value,
            "val_accuracy": best_result.val_accuracy,
        },
    )

    print(
        f"[train] dataset={dataset_name} "
        f"val_acc={best_result.val_accuracy:.6f} "
        f"L={best_result.num_layers} "
        f"L_fixed={best_result.num_fixed_layers} "
        f"C={best_result.c_value}"
    )

def main() -> None:
    """Loop over all filtered binary datasets and save the best hyperparameters for each."""
    config = load_config(CONFIG_PATH)
    dataset_names = list_binary_datasets(config)

    print(f"[train] found {len(dataset_names)} filtered binary datasets")

    for dataset_name in dataset_names:
        train_one_dataset(config, dataset_name)

if __name__ == "__main__":
    main()