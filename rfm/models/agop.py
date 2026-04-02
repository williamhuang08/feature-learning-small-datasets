import torch

from rfm.models.ntk import ntk_kernel_matrix
from rfm.utils.utils import matrix_sqrt_psd

def decision_function_from_original_inputs(
    x_orig: torch.Tensor,
    support_x_orig: torch.Tensor,
    dual_coef: torch.Tensor,
    intercept: torch.Tensor,
    M: torch.Tensor,
    num_layers: int,
    num_fixed_layers: int,
    eps: float,
) -> torch.Tensor:
    """
    Binary kernel SVM decision function evaluated at original input x.

    The classifier itself is trained on transformed features z = x M^{1/2},
    but the gradient is taken with respect to the original input x.

    Args:
        x_orig: (d,)
        support_x_orig: (n_support, d)
        dual_coef: (n_support,) signed dual coefficients from binary SVM
        intercept: scalar tensor
        M: (d, d) current RFM matrix
    """
    M_sqrt = matrix_sqrt_psd(M, eps)

    z = x_orig.unsqueeze(0) @ M_sqrt
    support_z = support_x_orig @ M_sqrt

    k_row = ntk_kernel_matrix(
        X1=z,
        X2=support_z,
        num_layers=num_layers,
        num_fixed_layers=num_fixed_layers,
        eps=eps,
    ).squeeze(0)

    return torch.sum(dual_coef * k_row) + intercept

def compute_agop_matrix(
    reference_x_orig: torch.Tensor,
    support_x_orig: torch.Tensor,
    dual_coef: torch.Tensor,
    intercept: torch.Tensor,
    M: torch.Tensor,
    num_layers: int,
    num_fixed_layers: int,
    eps: float,
) -> torch.Tensor:
    """
    Compute AGOP:
        M_star = (1 / n) sum_i grad_x f(x_i) grad_x f(x_i)^T

    Gradients are taken with respect to the original inputs.

    The result is symmetrized, sanitized, and normalized:
        M_star = M_star / (M_star.max() + eps)
    """
    d = reference_x_orig.shape[1]
    M_star = torch.zeros((d, d), dtype=reference_x_orig.dtype, device=reference_x_orig.device)

    for i in range(reference_x_orig.shape[0]):
        x = reference_x_orig[i].detach().clone().requires_grad_(True)

        value = decision_function_from_original_inputs(
            x_orig=x,
            support_x_orig=support_x_orig,
            dual_coef=dual_coef,
            intercept=intercept,
            M=M,
            num_layers=num_layers,
            num_fixed_layers=num_fixed_layers,
            eps=eps,
        )

        grad = torch.autograd.grad(value, x, create_graph=False, retain_graph=False)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        M_star = M_star + torch.outer(grad, grad)

    M_star = M_star / reference_x_orig.shape[0]
    M_star = M_star / (M_star.max() + eps)
    return M_star