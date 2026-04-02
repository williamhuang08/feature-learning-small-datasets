import torch

from tqdm import tqdm
from typing import Any
from itertools import product
from dataclasses import dataclass

from rfm.models.agop import compute_agop_matrix
from rfm.models.ntk import ntk_kernel_matrix
from rfm.models.svm import fit_precomputed_binary_svm
from rfm.utils.utils import accuracy_score, matrix_sqrt_psd, tensor_device, tensor_dtype

@dataclass
class SearchResult:
    """One hyperparameter search result."""
    num_iters: int
    num_layers: int
    num_fixed_layers: int
    c_value: float
    val_accuracy: float

class RFM:
    """
    Recursive Feature Machine with infinite-width ReLU NTK and binary C-SVM.

    This class keeps only the minimal fitted state required during the current run.
    The intended persistent artifact is the final matrix M, not the full classifier.
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_dim: int,
        num_iters: int,
        num_layers: int,
        num_fixed_layers: int,
        c_value: float,
    ) -> None:
        self.config = config
        self.input_dim = input_dim
        self.num_iters = num_iters
        self.num_layers = num_layers
        self.num_fixed_layers = num_fixed_layers
        self.c_value = c_value

        self.dtype = tensor_dtype(config)
        self.device = tensor_device(config)
        self.eps = float(config["eps"])

        self.M = torch.eye(input_dim, dtype=self.dtype, device=self.device)

        # Minimal fitted classifier state needed for prediction and AGOP.
        self.classes_: torch.Tensor | None = None
        self.support_x_orig: torch.Tensor | None = None
        self.dual_coef: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

    def transform(self, X_orig: torch.Tensor) -> torch.Tensor:
        """Apply z = x M^{1/2} using the current matrix M."""
        M_sqrt = matrix_sqrt_psd(self.M, self.eps)
        return X_orig @ M_sqrt

    def fit_svm(self, Z_train: torch.Tensor, y_train: torch.Tensor):
        """Fit binary C-SVM with the precomputed NTK Gram matrix."""
        return fit_precomputed_binary_svm(
            Z_train=Z_train,
            y_train=y_train,
            num_layers=self.num_layers,
            num_fixed_layers=self.num_fixed_layers,
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

        K = ntk_kernel_matrix(
            X1=Z,
            X2=support_Z,
            num_layers=self.num_layers,
            num_fixed_layers=self.num_fixed_layers,
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
        Fit one RFM model with fixed (num_iters, L, L', C).

        Convention:
            - num_iters = 0: fit once with M = I, no AGOP update
            - num_iters = T: perform T AGOP updates total

        AGOP is always averaged over the original training inputs.
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
                    num_layers=self.num_layers,
                    num_fixed_layers=self.num_fixed_layers,
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

        This is a search utility only. It does not restore the model to the
        best iterate after selection.
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
                    num_layers=self.num_layers,
                    num_fixed_layers=self.num_fixed_layers,
                    eps=self.eps,
                ).detach()

        if best_num_iters is None:
            raise RuntimeError("Validation search produced no result.")

        return best_num_iters, best_val_accuracy

def grid_search_rfm(
    config: dict[str, Any],
    X_train_orig: torch.Tensor,
    y_train: torch.Tensor,
    X_val_orig: torch.Tensor,
    y_val: torch.Tensor,
) -> SearchResult:
    """
    Grid search over:
        L in config["layer_list"]
        L' in {0, ..., L - 1}
        C in config["c_list"]

    For each fixed (L, L', C), run one trajectory up to max(iter_list),
    evaluate validation after each step, and pick the best iteration count
    from iter_list.
    """
    input_dim = X_train_orig.shape[1]
    best_result: SearchResult | None = None
    iter_candidates = [int(v) for v in config["iter_list"]]
    max_iters = max(iter_candidates)

    grid = []
    for num_layers, c_value in product(config["layer_list"], config["c_list"]):
        for num_fixed_layers in range(num_layers):
            grid.append((num_layers, num_fixed_layers, c_value))

    for num_layers, num_fixed_layers, c_value in tqdm(
        grid,
        total=len(grid),
        desc="Gridsearch",
    ):
        model = RFM(
            config=config,
            input_dim=input_dim,
            num_iters=max_iters,
            num_layers=num_layers,
            num_fixed_layers=num_fixed_layers,
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
                num_iters=best_num_iters,
                num_layers=num_layers,
                num_fixed_layers=num_fixed_layers,
                c_value=float(c_value),
                val_accuracy=float(val_acc),
            )

    if best_result is None:
        raise RuntimeError("Grid search produced no result.")

    return best_result