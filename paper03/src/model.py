from typing import Tuple

import torch


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


if __name__ == '__main__':
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data, target = datasets.make_blobs(
        n_samples=2048, n_features=2,
        centers=[
            [-2, -2], [2, 2],
            [-2, 2], [2, -2],
        ],
        cluster_std=1.4,
    )
    target = np.isin(target, [0, 1]) * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

    model = TorchMLP(n_epoch=500)
    model.fit(X_train, y_train > 0)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test > 0, y_pred > 0))
