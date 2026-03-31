import torch
import torch.autograd.functional as AF


def compute_agop(model, val_data):
    """Average (over n samples) of J^T J, where J is d(output)/d(input)"""
    model.eval()
    d_in = val_data.shape[1]
    agop = torch.zeros(d_in, d_in, device=val_data.device, dtype=val_data.dtype)
    n = val_data.shape[0]

    for i in range(n):
        x = val_data[i].detach().clone().requires_grad_(True)

        def f(inp):
            return model(inp.unsqueeze(0)).squeeze(0)

        jac = AF.jacobian(f, x)
        agop = agop + jac.T @ jac

    return agop / n
