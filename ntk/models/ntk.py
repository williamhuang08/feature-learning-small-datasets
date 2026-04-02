import math
import torch

def relu_covariance_step(
    sigma_12: torch.Tensor,
    diag_1: torch.Tensor,
    diag_2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One infinite-width ReLU covariance update.
    """
    norm = torch.sqrt(torch.clamp(torch.outer(diag_1, diag_2), min=eps))

    corr_margin = 1.0e-6
    corr = sigma_12 / norm
    corr = torch.clamp(corr, min=-1.0 + corr_margin, max=1.0 - corr_margin)

    theta = torch.arccos(corr)

    next_sigma_12 = (
        (corr * (math.pi - theta) + torch.sqrt(torch.clamp(1.0 - corr * corr, min=eps)))
        * norm
        / (2.0 * math.pi)
    )
    deriv = (math.pi - theta) / (2.0 * math.pi)

    next_diag_1 = 0.5 * diag_1
    next_diag_2 = 0.5 * diag_2

    return next_sigma_12, next_diag_1, next_diag_2, deriv

def ntk_kernel_matrix(
    X1: torch.Tensor,
    X2: torch.Tensor,
    num_layers: int,
    num_fixed_layers: int,
    eps: float,
) -> torch.Tensor:
    """
    Fully-connected infinite-width ReLU NTK with L total layers and L' bottom fixed layers.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if not (0 <= num_fixed_layers <= num_layers - 1):
        raise ValueError("num_fixed_layers must satisfy 0 <= L' <= L - 1.")

    sigma_12 = X1 @ X2.transpose(0, 1)
    diag_1 = torch.sum(X1 * X1, dim=1)
    diag_2 = torch.sum(X2 * X2, dim=1)
    H = torch.zeros_like(sigma_12)

    for layer_idx in range(num_layers):
        if num_fixed_layers <= layer_idx:
            H = H + sigma_12

        if layer_idx == num_layers - 1:
            return H

        sigma_12, diag_1, diag_2, deriv = relu_covariance_step(
            sigma_12=sigma_12,
            diag_1=diag_1,
            diag_2=diag_2,
            eps=eps,
        )
        H = H * deriv

    return H