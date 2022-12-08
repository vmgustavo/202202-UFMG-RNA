from typing import Tuple

import torch

from .torch_custom import get_silhouettes


class FeedForward(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...], output_size: int):
        super(FeedForward, self).__init__()

        sizes = (input_size, ) + hidden_sizes

        self.layers = torch.nn.ModuleList()
        for size_in, size_out in zip(sizes[::1], sizes[1::1]):
            self.layers.append(torch.nn.Linear(size_in, size_out))
            self.layers.append(torch.nn.modules.activation.Tanh())

        self.layers.append(torch.nn.Linear(sizes[-1], output_size))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        values = x
        for layer in self.layers:
            values = layer(values)

        return values

    def projection(self, x):
        values = x
        for layer in self.layers[:-2]:
            values = layer(values)

        return values


class TorchMLP:
    model_: FeedForward
    loss_: list

    def __init__(self, n_epoch: int, hidden_sizes: Tuple[int, ...] = (256, 256, 64)):
        self.n_epoch = n_epoch
        self.hidden_sizes = hidden_sizes

    def fit(self, X, y):
        self.model_ = FeedForward(input_size=X.shape[1], hidden_sizes=self.hidden_sizes, output_size=1)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01)

        tensor_x = torch.FloatTensor(X)
        tensor_y = torch.FloatTensor(y)

        self.loss_ = list()
        self.model_.train()
        for epoch in range(self.n_epoch):
            optimizer.zero_grad()
            loss = torch.nn.modules.loss.BCELoss()(
                self.model_(tensor_x).squeeze(),
                tensor_y
            )
            loss.backward()
            optimizer.step()
            self.loss_.append(loss.item())

        return self

    def predict(self, X, threshold: float = 0.5):
        return self.predict_proba(X) > threshold

    def predict_proba(self, X):
        self.model_.eval()
        return self.model_(torch.FloatTensor(X))


class SilWeightedBCELoss(torch.nn.Module):
    def __init__(self, thresh: float = 0.5):
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh
        self.history = list()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, projection: torch.Tensor):
        silhouettes = get_silhouettes(projection, targets).detach()
        silhouettes.requires_grad = False
        silhouette_weights = 1 - torch.pow(silhouettes, 2)
        self.history.append(torch.mean(silhouette_weights).item())

        loss = torch.nn.modules.loss.BCELoss(weight=silhouette_weights)(inputs, targets)
        return torch.mean(torch.multiply(loss, silhouette_weights)).detach()


class TorchMLPTopologyLoss(TorchMLP):
    model_: FeedForward
    loss_: list

    def __init__(self, n_epoch: int, hidden_sizes: Tuple[int, ...] = (256, 256, 64)):
        super(TorchMLPTopologyLoss, self).__init__(n_epoch, hidden_sizes)

    def fit(self, X, y):
        self.model_ = FeedForward(input_size=X.shape[1], hidden_sizes=self.hidden_sizes, output_size=1)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01)

        tensor_x = torch.FloatTensor(X)
        tensor_y = torch.FloatTensor(y)

        self.loss_ = list()
        self.model_.train()

        loss_function = SilWeightedBCELoss(thresh=0.5)
        for epoch in range(self.n_epoch):
            optimizer.zero_grad()
            loss = loss_function(
                inputs=self.model_(tensor_x).squeeze(),
                targets=tensor_y,
                projection=self.model_.projection(tensor_x)
            )
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            self.loss_.append(loss.item())

        return self

    def predict(self, X, threshold: float = 0.5):
        return self.predict_proba(X) > threshold

    def predict_proba(self, X):
        self.model_.eval()
        return self.model_(torch.FloatTensor(X))
