import os
import torch

def save_nn_checkpoint(path, model):
    os.makedirs(os.path.dirnmae(path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict()
    }, path)

    print(f"Model checkpoint saved to {path}")

def load_nn_checkpoint(path, model):
    ckpts = torch.load(path, weights_only=False, map_location='cpu')
    model.load_state_dict(ckpts['model_state_dict'])
    print(f"Model checkpoint loaded from {path}") 
