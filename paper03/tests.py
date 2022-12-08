from src.model import TorchMLP, TorchMLPTopologyLoss


def test_cluster_evaluate():
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

    model = TorchMLPTopologyLoss(n_epoch=1000)
    model.fit(X_train, y_train > 0)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test > 0, y_pred > 0))

    model = TorchMLP(n_epoch=500)
    model.fit(X_train, y_train > 0)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test > 0, y_pred > 0))