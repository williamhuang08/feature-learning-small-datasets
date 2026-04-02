import torch

def compute_layerwise_agop(model, reference_x: torch.Tensor) -> list[torch.Tensor]:
    """
    Compute AGOP with respect to each hidden layer input for a scalar network output.

    For hidden layer l with input feature h_{l-1}(x), compute:
        A_l = (1 / n) sum_i grad g_i grad g_i^T
    where
        g_i = f(x_i) is the scalar logit and grad = d f(x_i) / d h_{l-1}(x_i)

    Returns:
        layer_agops: list where entry l has shape (d_l, d_l), with
            d_0 = input_dim and d_l = h_dim for deeper hidden layers.
    """
    model.eval()
    reference_x = (
        reference_x
        .to(model.device, dtype=model.dtype)
        .detach()
        .requires_grad_(True)
    )

    with torch.enable_grad():
        layer_inputs, final_hidden = model.forward_features(reference_x)
        logits = model.output_linear(final_hidden).squeeze(-1)  # (n,)

        layer_agops: list[torch.Tensor] = []

        for layer_input in layer_inputs:
            d_layer = layer_input.shape[1]
            agop = torch.zeros(
                (d_layer, d_layer),
                dtype=reference_x.dtype,
                device=reference_x.device,
            )

            for sample_idx in range(reference_x.shape[0]):
                grad = torch.autograd.grad(
                    logits[sample_idx],
                    layer_input,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0][sample_idx]

                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                agop = agop + torch.outer(grad, grad)

            agop = agop / reference_x.shape[0]
            layer_agops.append(agop.detach())

    return layer_agops