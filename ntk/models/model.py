import torch

from tqdm import tqdm
from typing import Any
from itertools import product
from dataclasses import dataclass

from ntk.models.ntk import ntk_kernel_matrix
from ntk.models.svm import fit_precomputed_binary_svm
from rfm.utils.utils import accuracy_score, tensor_device, tensor_dtype

@dataclass
class SearchResult:
    """One hyperparameter search result."""
    num_layers: int
    num_fixed_layers: int
    c_value: float
    val_accuracy: float

class NTK:
    """
    Infinite-width ReLU NTK with binary C-SVM.
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_dim: int,
        num_layers: int,
        num_fixed_layers: int,
        c_value: float,
    ) -> None:
        self.config = config
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_fixed_layers = num_fixed_layers
        self.c_value = c_value

        self.dtype = tensor_dtype(config)
        self.device = tensor_device(config)
        self.eps = float(config["eps"])

        self.classes_: torch.Tensor | None = None
        self.support_x: torch.Tensor | None = None
        self.dual_coef: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

    def fit_svm(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Fit binary C-SVM with the precomputed NTK Gram matrix."""
        return fit_precomputed_binary_svm(
            X_train=X_train,
            y_train=y_train,
            num_layers=self.num_layers,
            num_fixed_layers=self.num_fixed_layers,
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

        K = ntk_kernel_matrix(
            X1=X,
            X2=self.support_x,
            num_layers=self.num_layers,
            num_fixed_layers=self.num_fixed_layers,
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

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> "NTK":
        """Fit one NTK model with fixed (L, L', C)."""
        if torch.unique(y_train).numel() != 2:
            raise ValueError("NTK only supports binary classification.")

        X_train = X_train.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device)

        clf = self.fit_svm(X_train, y_train)
        self.load_svm_state(clf, X_train)
        return self

def grid_search_ntk(
    config: dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> SearchResult:
    """
    Grid search over:
        L in config["layer_list"]
        L' in {0, ..., L - 1}
        C in config["c_list"]
    """
    input_dim = X_train.shape[1]
    best_result: SearchResult | None = None

    grid = []
    for num_layers, c_value in product(config["layer_list"], config["c_list"]):
        for num_fixed_layers in range(num_layers):
            grid.append((num_layers, num_fixed_layers, c_value))

    for num_layers, num_fixed_layers, c_value in tqdm(grid, total=len(grid), desc="Gridsearch"):
        model = NTK(
            config=config,
            input_dim=input_dim,
            num_layers=num_layers,
            num_fixed_layers=num_fixed_layers,
            c_value=float(c_value),
        )
        model.fit(X_train=X_train, y_train=y_train)
        val_acc = model.score(
            X_val.to(model.device, dtype=model.dtype),
            y_val.to(model.device),
        )

        if best_result is None or val_acc > best_result.val_accuracy:
            best_result = SearchResult(
                num_layers=num_layers,
                num_fixed_layers=num_fixed_layers,
                c_value=float(c_value),
                val_accuracy=float(val_acc),
            )

    if best_result is None:
        raise RuntimeError("Grid search produced no result.")

    return best_result