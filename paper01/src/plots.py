import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


def decision_boundary(
        model: (BaseEstimator, ClassifierMixin),
        X: np.array, y: np.array,
        savepath: str,
        steps: int = 100,
):
    value = np.ceil(np.max([X.max(), np.abs(X.min())]))
    grid = np.linspace(-1 * value, value, num=steps)
    grid_data = np.array([[x0, x1] for x0 in grid for x1 in grid])

    preds = model.predict(grid_data)

    plt.figure()
    plt.scatter(grid_data[:, 0], grid_data[:, 1], c=preds)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.contourf(
        grid, grid,
        preds.reshape(steps, steps),
        levels=1, alpha=.3, cmap="binary"
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.5)
    plt.tight_layout()
    plt.savefig(savepath)
