import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any
from itertools import product
from dataclasses import dataclass

from tqdm import tqdm

from nn_jf.utils.utils import tensor_device, tensor_dtype

@dataclass
class SearchResult:
    """One hyperparameter search result."""
    num_layers: int
    use_batch_norm: bool
    lr: float
    val_accuracy: float


class NN(nn.Module):
    """
    Feedforward ReLU network for binary classification with a scalar logit.

    Architecture:
        input -> [Linear -> (BatchNorm) -> ReLU] * num_layers -> Linear -> logit
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_dim: int,
        num_layers: int,
        h_dim: int,
        use_batch_norm: bool,
        lr: float,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.config = config
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.use_batch_norm = use_batch_norm
        self.lr = lr

        self.dtype = tensor_dtype(config)
        self.device = tensor_device(config)

        self.hidden_linears = nn.ModuleList()
        self.hidden_batch_norms = nn.ModuleList() if use_batch_norm else None
        self.output_linear = nn.Linear(h_dim, 1)
        self.activation = nn.ReLU()

        curr_dim = input_dim
        for _ in range(num_layers):
            self.hidden_linears.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                assert self.hidden_batch_norms is not None
                self.hidden_batch_norms.append(nn.BatchNorm1d(h_dim))
            curr_dim = h_dim

        self.to(device=self.device, dtype=self.dtype)

    def forward_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Run the hidden stack and return:
            - layer_inputs: input seen by each hidden Linear layer
            - final_hidden: hidden representation before the output layer
        """
        layer_inputs: list[torch.Tensor] = []
        h = x

        for layer_idx, linear in enumerate(self.hidden_linears):
            layer_inputs.append(h)

            h = linear(h)
            if self.use_batch_norm:
                assert self.hidden_batch_norms is not None
                h = self.hidden_batch_norms[layer_idx](h)
            h = self.activation(h)

        return layer_inputs, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar logits of shape (n,)."""
        _, h = self.forward_features(x)
        return self.output_linear(h).squeeze(-1)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "NN":
        """Fit the network for a fixed number of epochs."""
        X_train = X_train.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device, dtype=self.dtype)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        for _ in range(int(self.config["num_epochs"])):
            self.train()
            optimizer.zero_grad()
            logits = self(X_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict binary class labels in {0, 1}."""
        self.eval()
        X = X.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            logits = self(X)
            return (logits >= 0).long()

    def score(self, X: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute classification accuracy."""
        y_pred = self.predict(X)
        y_true = y_true.to(self.device).long()
        return float((y_pred == y_true).float().mean().item())

    def best_val_accuracy(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> float:
        """
        Train the model and track the best validation accuracy over epochs.
        """
        X_train = X_train.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device, dtype=self.dtype)
        X_val = X_val.to(self.device, dtype=self.dtype)
        y_val = y_val.to(self.device).long()

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        best_val_accuracy = float("-inf")

        for _ in range(int(self.config["num_epochs"])):
            self.train()
            optimizer.zero_grad()
            train_logits = self(X_train)
            train_loss = loss_fn(train_logits, y_train)
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits = self(X_val)
                val_pred = (val_logits >= 0).long()
                val_accuracy = float((val_pred == y_val).float().mean().item())

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = float(val_accuracy)

        return best_val_accuracy

def grid_search_nn(
    config: dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> SearchResult:
    """
    Grid search over:
        num_layers in config["layer_list"]
        use_batch_norm in config["batch_norm_list"]
        lr in config["lr_list"]
    """
    input_dim = X_train.shape[1]
    best_result: SearchResult | None = None

    grid = list(
        product(
            config["lr_list"],
            config["batch_norm_list"],
            config["layer_list"],
        )
    )

    for lr, use_batch_norm, num_layers in tqdm(
        grid,
        total=len(grid),
        desc="Gridsearch",
    ):
        model = NN(
            config=config,
            input_dim=input_dim,
            num_layers=int(num_layers),
            h_dim=int(config["hidden_dim"]),
            use_batch_norm=bool(use_batch_norm),
            lr=float(lr),
        )

        val_accuracy = model.best_val_accuracy(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        if best_result is None or val_accuracy > best_result.val_accuracy:
            best_result = SearchResult(
                num_layers=int(num_layers),
                use_batch_norm=bool(use_batch_norm),
                lr=float(lr),
                val_accuracy=float(val_accuracy),
            )

    if best_result is None:
        raise RuntimeError("Grid search produced no result.")

    return best_result