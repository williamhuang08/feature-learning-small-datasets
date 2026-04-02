import torch

from sklearn.svm import SVC

from rfm.models.ntk import ntk_kernel_matrix

def fit_precomputed_binary_svm(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    num_layers: int,
    num_fixed_layers: int,
    c_value: float,
    eps: float,
) -> SVC:
    """
    Fit a binary C-SVM using the NTK Gram matrix on transformed features.

    Args:
        Z_train: transformed inputs, shape (n_train, d)
        y_train: binary labels, shape (n_train,)
    """
    K_train = ntk_kernel_matrix(
        X1=Z_train,
        X2=Z_train,
        num_layers=num_layers,
        num_fixed_layers=num_fixed_layers,
        eps=eps,
    )

    clf = SVC(kernel="precomputed", C=c_value, cache_size=100000)
    clf.fit(K_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    return clf