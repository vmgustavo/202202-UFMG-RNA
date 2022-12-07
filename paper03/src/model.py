import torch


class FeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.activation = torch.nn.modules.activation.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        activation = self.activation(hidden)
        output = self.fc2(activation)
        output = self.sigmoid(output)
        return output


class TorchMLP:
    model_: FeedForward
    loss_: list

    def __init__(self, n_epoch: int):
        self.n_epoch = n_epoch
        self.criterion = torch.nn.modules.loss.BCELoss()

    def fit(self, X, y):
        self.model_ = FeedForward(input_size=X.shape[1], hidden_size=100)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01)

        tensor_x = torch.FloatTensor(X)
        tensor_y = torch.FloatTensor(y)

        self.loss_ = list()
        self.model_.train()
        for epoch in range(self.n_epoch):
            optimizer.zero_grad()
            loss = self.criterion(self.model_(tensor_x).squeeze(), tensor_y)
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
