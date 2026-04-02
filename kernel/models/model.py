import torch

from tqdm import tqdm
from typing import Any
from itertools import product
from dataclasses import dataclass

from kernel.models.kernels import kernel_matrix
from kernel.models.svm import fit_precomputed_binary_svm
from kernel.utils.utils import accuracy_score, tensor_device, tensor_dtype


@dataclass
class SearchResult:
    """One hyperparameter search result."""
    kernel: str
    kernel_params: dict[str, float | int]
    c_value: float
    val_accuracy: float

class KernelSVM:
    """Kernel SVM with binary C-SVM and a selectable precomputed kernel."""

    def __init__(
        self,
        config: dict[str, Any],
        input_dim: int,
        kernel_name: str,
        kernel_params: dict[str, float | int],
        c_value: float,
    ) -> None:
        self.config = config
        self.input_dim = input_dim
        self.kernel_name = kernel_name
        self.kernel_params = kernel_params
        self.c_value = c_value

        self.dtype = tensor_dtype(config)
        self.device = tensor_device(config)
        self.eps = float(config["eps"])

        self.classes_: torch.Tensor | None = None
        self.support_x: torch.Tensor | None = None
        self.dual_coef: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

    def fit_svm(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Fit binary C-SVM with the selected precomputed Gram matrix."""
        return fit_precomputed_binary_svm(
            X_train=X_train,
            y_train=y_train,
            kernel_name=self.kernel_name,
            kernel_params=self.kernel_params,
            c_value=self.c_value,
            eps=self.eps,
        )

    def load_svm_state(self, clf, X_train: torch.Tensor) -> None:
        """Extract only the state needed for prediction."""
        support_indices = torch.tensor(clf.support_, dtype=torch.long, device=self.device)

        self.classes_ = torch.tensor(clf.classes_, dtype=torch.long, device=self.device)
        self.support_x = X_train[support_indices].detach().clone()
        self.dual_coef = torch.tensor(clf.dual_coef_[0], dtype=self.dtype, device=self.device)
        self.intercept = torch.tensor(clf.intercept_[0], dtype=self.dtype, device=self.device)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Binary decision function."""
        if self.support_x is None or self.dual_coef is None or self.intercept is None:
            raise RuntimeError("Model is not fitted.")

        K = kernel_matrix(
            kernel_name=self.kernel_name,
            X1=X,
            X2=self.support_x,
            params=self.kernel_params,
            eps=self.eps,
        )
        return K @ self.dual_coef + self.intercept

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict binary labels."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        scores = self.decision_function(X)
        pred_pos = (scores >= 0).long()
        return self.classes_[pred_pos]

    def score(self, X: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute classification accuracy."""
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> "KernelSVM":
        """Fit one kernel SVM model with fixed hyperparameters."""
        if torch.unique(y_train).numel() != 2:
            raise ValueError("KernelSVM only supports binary classification.")

        X_train = X_train.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device)

        clf = self.fit_svm(X_train, y_train)
        self.load_svm_state(clf, X_train)
        return self

# kernel/models/model.py
def build_kernel_grid(config: dict[str, Any]) -> list[tuple[dict[str, float | int], float]]:
    """Build the hyperparameter grid for the selected kernel."""
    kernel_name = config["kernel"]

    if kernel_name == "gaussian":
        return [
            ({"gamma": float(gamma)}, float(c_value))
            for gamma, c_value in product(config["gamma_list"], config["c_list"])
        ]

    if kernel_name == "laplace":
        return [
            ({"gamma": float(gamma)}, float(c_value))
            for gamma, c_value in product(config["gamma_list"], config["c_list"])
        ]

    if kernel_name == "polynomial":
        return [
            (
                {
                    "degree": int(degree),
                    "gamma": float(gamma),
                    "coef0": float(coef0),
                },
                float(c_value),
            )
            for degree, gamma, coef0, c_value in product(
                config["degree_list"],
                config["gamma_list"],
                config["coef0_list"],
                config["c_list"],
            )
        ]

    if kernel_name == "ntk":
        grid: list[tuple[dict[str, float | int], float]] = []
        for num_layers, c_value in product(config["layer_list"], config["c_list"]):
            for num_fixed_layers in range(int(num_layers)):
                grid.append(
                    (
                        {
                            "num_layers": int(num_layers),
                            "num_fixed_layers": int(num_fixed_layers),
                        },
                        float(c_value),
                    )
                )
        return grid

    raise ValueError(f"Unsupported kernel: {kernel_name}")

def grid_search_kernel(
    config: dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> SearchResult:
    """Grid search over the hyperparameters of the selected kernel."""
    input_dim = X_train.shape[1]
    best_result: SearchResult | None = None
    kernel_name = config["kernel"]
    grid = build_kernel_grid(config)

    for kernel_params, c_value in tqdm(grid, total=len(grid), desc="Gridsearch"):
        model = KernelSVM(
            config=config,
            input_dim=input_dim,
            kernel_name=kernel_name,
            kernel_params=kernel_params,
            c_value=float(c_value),
        )
        model.fit(X_train=X_train, y_train=y_train)
        val_acc = model.score(
            X_val.to(model.device, dtype=model.dtype),
            y_val.to(model.device),
        )

        if best_result is None or val_acc > best_result.val_accuracy:
            best_result = SearchResult(
                kernel=kernel_name,
                kernel_params=kernel_params,
                c_value=float(c_value),
                val_accuracy=float(val_acc),
            )

    if best_result is None:
        raise RuntimeError("Grid search produced no result.")

    return best_result