import os
import json
from time import time
from itertools import product

import typer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from datasets import get_linear, get_blobs, get_moons
from models import ELMClassifier, ELMRegClassifier, ELMPCAClassifier

app = typer.Typer()


@app.command()
def experiment(n_obs: int = 1024, neurons_steps: int = 10):
    executor(n_obs, neurons_steps)


@app.command()
def repeat(n: int, n_obs: int = 1024, neurons_steps: int = 10):
    for _ in range(n):
        executor(n_obs, neurons_steps)


def executor(n_obs: int = 1024, neurons_steps: int = 10):
    datasets = [
        (get_linear(n_obs, 2), "linear"),
        (get_blobs(n_obs, 2), "blobs"),
        (get_moons(n_obs), "moons")
    ]

    for dataset, dataset_name in datasets:
        num_neurons = (2 ** elem for elem in range(neurons_steps + 1, 1, -1))

        models = [
            (ELMClassifier, "ELM"),
            (ELMRegClassifier, "ELMReg"),
            (ELMPCAClassifier, "ELMPCA"),
        ]

        for (model_cls, model_name), neurons in product(models, num_neurons):

            model = model_cls(neurons=neurons)
            X_train, X_test, y_train, y_test = train_test_split(*dataset, test_size=.3)

            st = time()
            model.fit(X_train, y_train)
            time_to_fit = time() - st

            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            confmat = confusion_matrix(y_test > 0, pred_test)

            ##############################################################################
            print(dataset_name, model.__class__.__name__, f"{neurons:04d}")
            print(confmat, end=" ")
            print(accuracy_score(y_train > 0, model.predict(X_train)), end=" ")
            print(accuracy_score(y_test > 0, pred_test))
            ##############################################################################

            tn, fp, fn, tp = confmat.ravel()

            results = {
                "dataset": dataset_name,
                "model": model_name,
                "neurons": neurons,
                "evaluation": {
                    "acc_train": accuracy_score(y_train > 0, pred_train),
                    "acc_test": accuracy_score(y_test > 0, pred_test),
                    "time_to_fit": time_to_fit,
                    "confmat": {
                        "true_neg": int(tn), "false_pos": int(fp),
                        "false_neg": int(fn), "true_pos": int(tp),
                    },
                },
            }

            if not os.path.exists("results"):
                os.mkdir("results")

            with open(f"results/{time():.0f}.json", "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    app()
