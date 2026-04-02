import torch

from sklearn.svm import SVC

from kernel.models.kernels import kernel_matrix

def fit_precomputed_binary_svm(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    kernel_name: str,
    kernel_params: dict[str, float | int],
    c_value: float,
    eps: float,
) -> SVC:
    """Fit a binary C-SVM using a precomputed Gram matrix."""
    K_train = kernel_matrix(
        kernel_name=kernel_name,
        X1=X_train,
        X2=X_train,
        params=kernel_params,
        eps=eps,
    )

    clf = SVC(kernel="precomputed", C=c_value, cache_size=100000)
    clf.fit(K_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    return clf