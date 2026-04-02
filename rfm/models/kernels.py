import math
import torch

def gaussian_kernel_matrix(
    X1: torch.Tensor,
    X2: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Gaussian / RBF kernel matrix."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    x1_sq = torch.sum(X1 * X1, dim=1, keepdim=True)
    x2_sq = torch.sum(X2 * X2, dim=1, keepdim=True).transpose(0, 1)
    sqdist = x1_sq + x2_sq - 2.0 * (X1 @ X2.transpose(0, 1))
    sqdist = torch.clamp(sqdist, min=0.0)
    return torch.exp(-gamma * sqdist)

def laplace_kernel_matrix(
    X1: torch.Tensor,
    X2: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Laplace kernel matrix: exp(-gamma * ||x - z||_1)."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    l1dist = torch.sum(torch.abs(X1[:, None, :] - X2[None, :, :]), dim=2)
    return torch.exp(-gamma * l1dist)

def polynomial_kernel_matrix(
    X1: torch.Tensor,
    X2: torch.Tensor,
    degree: int,
    gamma: float,
    coef0: float,
) -> torch.Tensor:
    """Polynomial kernel matrix: (gamma * <x, z> + coef0)^degree."""
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    return (gamma * (X1 @ X2.transpose(0, 1)) + coef0) ** degree

def relu_covariance_step(
    sigma_12: torch.Tensor,
    diag_1: torch.Tensor,
    diag_2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One infinite-width ReLU covariance update."""
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
    """Fully-connected infinite-width ReLU NTK."""
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

def kernel_matrix(
    kernel_name: str,
    X1: torch.Tensor,
    X2: torch.Tensor,
    params: dict[str, float | int],
    eps: float,
) -> torch.Tensor:
    """Dispatch to the selected kernel matrix implementation."""
    if kernel_name == "gaussian":
        return gaussian_kernel_matrix(
            X1=X1,
            X2=X2,
            gamma=float(params["gamma"]),
        )

    if kernel_name == "laplace":
        return laplace_kernel_matrix(
            X1=X1,
            X2=X2,
            gamma=float(params["gamma"]),
        )

    if kernel_name == "polynomial":
        return polynomial_kernel_matrix(
            X1=X1,
            X2=X2,
            degree=int(params["degree"]),
            gamma=float(params["gamma"]),
            coef0=float(params["coef0"]),
        )

    if kernel_name == "ntk":
        return ntk_kernel_matrix(
            X1=X1,
            X2=X2,
            num_layers=int(params["num_layers"]),
            num_fixed_layers=int(params["num_fixed_layers"]),
            eps=eps,
        )

    raise ValueError(f"Unsupported kernel: {kernel_name}")