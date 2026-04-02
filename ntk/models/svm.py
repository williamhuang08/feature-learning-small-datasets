import torch

from sklearn.svm import SVC

from ntk.models.ntk import ntk_kernel_matrix

def fit_precomputed_binary_svm(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_layers: int,
    num_fixed_layers: int,
    c_value: float,
    eps: float,
) -> SVC:
    """
    Fit a binary C-SVM using the NTK Gram matrix.
    """
    K_train = ntk_kernel_matrix(
        X1=X_train,
        X2=X_train,
        num_layers=num_layers,
        num_fixed_layers=num_fixed_layers,
        eps=eps,
    )

    clf = SVC(kernel="precomputed", C=c_value, cache_size=100000)
    clf.fit(K_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    return clf