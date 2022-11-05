import os
import json
import logging
import hashlib
import warnings
from time import time

import click
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from datasets import alldts
from metrics import cluster_evaluate

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


def executor():
    selection = {"cred_aus", "cred_ger", "breast_coimbra", "sonar", "heart"}
    datasets = [(k, v) for k, v in alldts().items() if k in selection]
    alphas = [0] + list(np.logspace(-1, 1/2, num=20))

    for dataset_name, (data, target) in tqdm(datasets):
        for alpha in alphas:
            X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, stratify=target, test_size=.3)

            model = MLPClassifier(
                hidden_layer_sizes=(int(2**10), ), activation="tanh", solver="adam",
                alpha=alpha, beta_1=0.9, beta_2=0.999,
                max_iter=1024,
                verbose=False, shuffle=False,
                early_stopping=False, validation_fraction=0.1,
                n_iter_no_change=512, tol=1e-6,
                epsilon=1e-8, learning_rate="constant",
            )

            model.fit(X_train, y_train)

            results = dict({
                "dataset": dataset_name,
                "alpha": alpha,
                "acc_train": accuracy_score(y_pred=model.predict(X_train), y_true=y_train),
                "acc_test": accuracy_score(y_pred=model.predict(X_test), y_true=y_test),
                "best_loss": model.best_loss_,
                "iterations": model.n_iter_
            }, **cluster_evaluate(X=np.tanh(X_train @ model.coefs_[0]), labels=y_train))

            if not os.path.exists("results"):
                os.mkdir("results")

            fname = str(time()) + str(alpha) + dataset_name
            with open(f"results/{hashlib.sha256(fname.encode()).hexdigest()}.json", "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    repeat()
