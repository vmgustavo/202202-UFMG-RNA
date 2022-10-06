from time import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from datasets import get_linear, get_blobs, get_moons
from models import ELMClassifier, ELMRegClassifier, ELMPCAClassifier


def main():
    n_obs = 2048
    n_feats = 2
    dataset_opts = (
        (f"linear__{n_obs}_obs__{n_feats}_feats", get_linear(n_obs, n_feats)),
        (f"blobs__{n_obs}_obs__{n_feats}_feats", get_blobs(n_obs, n_feats)),
        (f"moons__{n_obs}_obs", get_moons(n_obs))
    )

    results = list()

    for dataset_name, dataset in tqdm(dataset_opts, desc="dataset"):
        X_train, X_test, y_train, y_test = train_test_split(*dataset, test_size=.3)

        for neurons in tqdm(range(2, 256, 10), desc="neurons"):
            model_opts = (
                ELMClassifier(neurons=neurons),
                ELMRegClassifier(neurons=neurons),
                ELMPCAClassifier(neurons=neurons)
            )

            for model in model_opts:
                st = time()
                model.fit(X_train, y_train)
                time_to_fit = time() - st

                _, bias, variance = bias_variance_decomp(
                    model,
                    X_train, y_train > 0,
                    X_test, y_test > 0,
                    num_rounds=50
                )

                results.append({
                    "dataset": dataset_name,
                    "model": model.__class__.__name__,
                    "neurons": neurons,
                    "evaluation": {
                        "training": accuracy_score(
                            y_true=y_train > 0,
                            y_pred=model.predict(X_train).reshape(-1) > 0,
                        ),
                        "testing": accuracy_score(
                            y_true=y_test > 0,
                            y_pred=model.predict(X_test).reshape(-1) > 0,
                        ),
                        "bias": bias,
                        "variance": variance,
                        "time_to_fit": time_to_fit
                    },
                })

    (
        pd.json_normalize(results)
        .to_csv(f"results__{int(datetime.now().timestamp())}.csv", index=False)
    )


if __name__ == '__main__':
    main()
