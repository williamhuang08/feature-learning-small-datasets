import torch

def compute_layerwise_nfm(model) -> list[torch.Tensor]:
    """
    Compute layerwise Neural Feature Matrices for the hidden Linear layers.

    For hidden layer l with weight W_l of shape (h_dim, d_l), compute:
        N_l = W_l^T W_l

    Returns:
        layer_nfms: list where entry l has shape (d_l, d_l).
    """
    layer_nfms: list[torch.Tensor] = []

    for linear in model.hidden_linears:
        weight = linear.weight.detach()  # (out_dim, in_dim)
        nfm = weight.transpose(0, 1) @ weight
        layer_nfms.append(nfm)

    return layer_nfms