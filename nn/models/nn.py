import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, num_layers, input_dim, h_dim, output_dim, use_batch_norm):
        super(NN, self).__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.h_dim = h_dim 
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        self.model = nn.ModuleList()

        curr_dim = self.input_dim
        for l in self.num_layers:
            self.layers.append(nn.Linear(curr_dim, self.h_dim))
            if self.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(self.h_dim))
            self.layers.append(F.relu())
            curr_dim = self.h_dim
        
        self.model.append(nn.Linear(self.h_dim, self.output_dim))

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


