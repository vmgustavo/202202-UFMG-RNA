import os
import json
import logging
import hashlib
import warnings
from time import time
from dataclasses import dataclass

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.model import TorchMLP

from src.datasets import alldts
from src.metrics import cluster_evaluate

logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@click.command()
@click.option('-n', type=int, required=True)
def repeat(n):
    for _ in range(n):
        executor()


@dataclass
class ModelResults:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    hidden_layer_sizes: tuple
    reg_alpha: float
    projection: np.array
    pred_train: np.array
    pred_test: np.array


def exec_nn(data: pd.DataFrame, target: pd.Series, hidden_layer_sizes: tuple, reg_alpha: float):
    X_train, X_test, y_train, y_test = train_test_split(
        data.values, target.values,
        stratify=target, test_size=.3
    )

    model = TorchMLP(n_epoch=1024, hidden_sizes=hidden_layer_sizes)
    model.fit(X_train, y_train)
    projection = (
        model.model_.projection(torch.Tensor(X_train))
        .detach()
        .numpy()
        .astype("float")
    )

    return ModelResults(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        hidden_layer_sizes=hidden_layer_sizes, reg_alpha=reg_alpha,
        projection=projection,
        pred_train=model.predict(X_train).numpy(), pred_test=model.predict(X_test).numpy(),
    )


def executor():
    selection = {
        "cred_aus", "cred_ger",
        "breast_coimbra", "sonar", "heart",
        "synth_linear", "synth_blobs", "synth_moons",
    }
    datasets = [(k, v) for k, v in alldts().items() if k in selection]
    alphas = np.logspace(-1, 1, 16)

    for dataset_name, (data, target) in datasets:
        for alpha in alphas:
            logging.info(f"dataset: {dataset_name:>15} | alpha: {alpha:02.02f}")
            res = exec_nn(
                data, target,
                hidden_layer_sizes=(128, 128, 128, 128, 128), reg_alpha=alpha,
            )

            results = dict(
                {
                    "dataset": dataset_name,
                    "alpha": alpha,
                    "acc_train": accuracy_score(y_pred=res.pred_train, y_true=res.y_train),
                    "acc_test": accuracy_score(y_pred=res.pred_test, y_true=res.y_test),
                },
                **cluster_evaluate(X=res.projection, labels=res.y_train),
                **{f"orig_{k}": v for k, v in cluster_evaluate(X=res.X_train, labels=res.y_train).items()}
            )

            if not os.path.exists("results"):
                os.mkdir("results")

            fname = str(time()) + str(alpha) + dataset_name
            with open(f"results/{hashlib.sha256(fname.encode()).hexdigest()}.json", "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    repeat()
