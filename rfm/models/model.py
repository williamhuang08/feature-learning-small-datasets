# rfm/models/model.py
import torch

from tqdm import tqdm
from typing import Any
from itertools import product
from dataclasses import dataclass

from rfm.models.agop import compute_agop_matrix
from rfm.models.kernels import kernel_matrix
from rfm.models.svm import fit_precomputed_binary_svm
from rfm.utils.utils import accuracy_score, matrix_sqrt_psd, tensor_device, tensor_dtype

@dataclass
class SearchResult:
    """One hyperparameter search result."""
    kernel: str
    num_iters: int
    kernel_params: dict[str, float | int]
    c_value: float
    val_accuracy: float

class RFM:
    """
    Recursive Feature Machine with selectable kernel and binary C-SVM.

    The persistent artifact is the final matrix M, not the full classifier.
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_dim: int,
        kernel_name: str,
        num_iters: int,
        kernel_params: dict[str, float | int],
        c_value: float,
    ) -> None:
        self.config = config
        self.input_dim = input_dim
        self.kernel_name = kernel_name
        self.num_iters = num_iters
        self.kernel_params = kernel_params
        self.c_value = c_value

        self.dtype = tensor_dtype(config)
        self.device = tensor_device(config)
        self.eps = float(config["eps"])

        self.M = torch.eye(input_dim, dtype=self.dtype, device=self.device)

        self.classes_: torch.Tensor | None = None
        self.support_x_orig: torch.Tensor | None = None
        self.dual_coef: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

    def transform(self, X_orig: torch.Tensor) -> torch.Tensor:
        """Apply z = x M^{1/2} using the current matrix M."""
        M_sqrt = matrix_sqrt_psd(self.M, self.eps)
        return X_orig @ M_sqrt

    def fit_svm(self, Z_train: torch.Tensor, y_train: torch.Tensor):
        """Fit binary C-SVM with the selected precomputed Gram matrix."""
        return fit_precomputed_binary_svm(
            Z_train=Z_train,
            y_train=y_train,
            kernel_name=self.kernel_name,
            kernel_params=self.kernel_params,
            c_value=self.c_value,
            eps=self.eps,
        )

    def load_svm_state(self, clf, X_train_orig: torch.Tensor) -> None:
        """Extract only the state needed for prediction and AGOP."""
        support_indices = torch.tensor(clf.support_, dtype=torch.long, device=self.device)

        self.classes_ = torch.tensor(clf.classes_, dtype=torch.long, device=self.device)
        self.support_x_orig = X_train_orig[support_indices].detach().clone()
        self.dual_coef = torch.tensor(clf.dual_coef_[0], dtype=self.dtype, device=self.device)
        self.intercept = torch.tensor(clf.intercept_[0], dtype=self.dtype, device=self.device)

    def decision_function(self, X_orig: torch.Tensor) -> torch.Tensor:
        """Binary decision function on original inputs."""
        if self.support_x_orig is None or self.dual_coef is None or self.intercept is None:
            raise RuntimeError("Model is not fitted.")

        Z = self.transform(X_orig)
        support_Z = self.transform(self.support_x_orig)

        K = kernel_matrix(
            kernel_name=self.kernel_name,
            X1=Z,
            X2=support_Z,
            params=self.kernel_params,
            eps=self.eps,
        )
        return K @ self.dual_coef + self.intercept

    def predict(self, X_orig: torch.Tensor) -> torch.Tensor:
        """Predict binary labels on original inputs."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        scores = self.decision_function(X_orig)
        pred_pos = (scores >= 0).long()
        return self.classes_[pred_pos]

    def score(self, X_orig: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute classification accuracy."""
        y_pred = self.predict(X_orig)
        return accuracy_score(y_true, y_pred)

    def fit(
        self,
        X_train_orig: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "RFM":
        """
        Fit one RFM model with fixed hyperparameters.

        Convention:
            - num_iters = 0: fit once with M = I, no AGOP update
            - num_iters = T: perform T AGOP updates total
        """
        if torch.unique(y_train).numel() != 2:
            raise ValueError("RFM only supports binary classification.")

        X_train_orig = X_train_orig.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device)

        for iter_idx in range(self.num_iters + 1):
            Z_train = self.transform(X_train_orig)

            clf = self.fit_svm(Z_train, y_train)
            self.load_svm_state(clf, X_train_orig)

            if iter_idx < self.num_iters:
                self.M = compute_agop_matrix(
                    reference_x_orig=X_train_orig,
                    support_x_orig=self.support_x_orig,
                    dual_coef=self.dual_coef,
                    intercept=self.intercept,
                    M=self.M,
                    kernel_name=self.kernel_name,
                    kernel_params=self.kernel_params,
                    eps=self.eps,
                ).detach()

        return self

    def fit_and_select_num_iters(
        self,
        X_train_orig: torch.Tensor,
        y_train: torch.Tensor,
        X_val_orig: torch.Tensor,
        y_val: torch.Tensor,
        iter_candidates: list[int],
    ) -> tuple[int, float]:
        """
        Run one RFM trajectory up to max(iter_candidates), evaluate validation
        after each SVM fit, and return the best iteration count from
        iter_candidates together with its validation accuracy.
        """
        if torch.unique(y_train).numel() != 2:
            raise ValueError("RFM only supports binary classification.")

        iter_candidates = sorted(set(int(v) for v in iter_candidates))
        if iter_candidates[0] < 0:
            raise ValueError("iter_candidates must be non-negative.")

        max_iters = iter_candidates[-1]
        candidate_set = set(iter_candidates)

        X_train_orig = X_train_orig.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device)
        X_val_orig = X_val_orig.to(self.device, dtype=self.dtype)
        y_val = y_val.to(self.device)

        best_num_iters: int | None = None
        best_val_accuracy = float("-inf")

        for iter_idx in range(max_iters + 1):
            Z_train = self.transform(X_train_orig)

            clf = self.fit_svm(Z_train, y_train)
            self.load_svm_state(clf, X_train_orig)

            if iter_idx in candidate_set:
                val_accuracy = self.score(X_val_orig, y_val)

                if val_accuracy > best_val_accuracy:
                    best_num_iters = iter_idx
                    best_val_accuracy = float(val_accuracy)

            if iter_idx < max_iters:
                self.M = compute_agop_matrix(
                    reference_x_orig=X_train_orig,
                    support_x_orig=self.support_x_orig,
                    dual_coef=self.dual_coef,
                    intercept=self.intercept,
                    M=self.M,
                    kernel_name=self.kernel_name,
                    kernel_params=self.kernel_params,
                    eps=self.eps,
                ).detach()

        if best_num_iters is None:
            raise RuntimeError("Validation search produced no result.")

        return best_num_iters, best_val_accuracy

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

def grid_search_rfm(
    config: dict[str, Any],
    X_train_orig: torch.Tensor,
    y_train: torch.Tensor,
    X_val_orig: torch.Tensor,
    y_val: torch.Tensor,
) -> SearchResult:
    """
    Grid search over the selected kernel hyperparameters and C.

    For each fixed kernel configuration and C, run one trajectory up to
    max(iter_list), evaluate validation after each step, and pick the best
    iteration count from iter_list.
    """
    input_dim = X_train_orig.shape[1]
    best_result: SearchResult | None = None
    kernel_name = config["kernel"]
    iter_candidates = [int(v) for v in config["iter_list"]]
    max_iters = max(iter_candidates)
    grid = build_kernel_grid(config)

    for kernel_params, c_value in tqdm(grid, total=len(grid), desc="Gridsearch"):
        model = RFM(
            config=config,
            input_dim=input_dim,
            kernel_name=kernel_name,
            num_iters=max_iters,
            kernel_params=kernel_params,
            c_value=float(c_value),
        )

        best_num_iters, val_acc = model.fit_and_select_num_iters(
            X_train_orig=X_train_orig,
            y_train=y_train,
            X_val_orig=X_val_orig,
            y_val=y_val,
            iter_candidates=iter_candidates,
        )

        if best_result is None or val_acc > best_result.val_accuracy:
            best_result = SearchResult(
                kernel=kernel_name,
                num_iters=best_num_iters,
                kernel_params=kernel_params,
                c_value=float(c_value),
                val_accuracy=float(val_acc),
            )

    if best_result is None:
        raise RuntimeError("Grid search produced no result.")

    return best_result