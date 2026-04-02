# rfm/models/svm.py
import torch

from sklearn.svm import SVC

from rfm.models.kernels import kernel_matrix


def fit_precomputed_binary_svm(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    kernel_name: str,
    kernel_params: dict[str, float | int],
    c_value: float,
    eps: float,
) -> SVC:
    """Fit a binary C-SVM using a precomputed Gram matrix."""
    K_train = kernel_matrix(
        kernel_name=kernel_name,
        X1=Z_train,
        X2=Z_train,
        params=kernel_params,
        eps=eps,
    )

    clf = SVC(kernel="precomputed", C=c_value, cache_size=100000)
    clf.fit(K_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    return clf