from io import BytesIO

import requests
import numpy as np
import pandas as pd
from sklearn import datasets


def get_linear(n_obs: int, n_feats: int):
    data, target = datasets.make_blobs(
        n_samples=n_obs, n_features=n_feats,
        centers=[[-2, -2], [2, 2]],
        cluster_std=1.5,
    )
    target = (target == 1) * 2 - 1
    return data, target


def get_blobs(n_obs: int, n_feats: int):
    data, target = datasets.make_blobs(
        n_samples=n_obs, n_features=n_feats,
        centers=[
            [-2, -2], [2, 2],
            [-2, 2], [2, -2],
        ],
        cluster_std=1.2,
    )
    target = np.isin(target, [0, 1]) * 2 - 1
    return data, target


def get_moons(n_obs: int):
    data, target = datasets.make_moons(
        n_samples=n_obs, noise=0.2,
    )

    target = (target == 1) * 2 - 1
    data = data - [data[:, 0].mean(), data[:, 1].mean()]
    return data, target


def get_ilpd():
    """ ILPD (Indian Liver Patient Dataset) Data Set
    Data Set Information: This data set contains 416 liver patient records and
    167 non liver patient records. The data set was collected from north east of
    Andhra Pradesh, India. Selector is a class label used to divide into
    groups (liver patient or not). This data set contains 441 male patient
    records and 142 female patient records.

    Any patient whose age exceeded 89 is listed as being of age "90".

    Attribute Information:
        1. Age: Age of the patient
        2. Gender: Gender of the patient
        3. TB: Total Bilirubin
        4. DB: Direct Bilirubin
        5. Alkphos: Alkaline Phosphotase
        6. Sgpt: Alamine Aminotransferase
        7. Sgot: Aspartate Aminotransferase
        8. TP: Total Protiens
        9. ALB: Albumin
        10. A/G: Ratio Albumin and Globulin Ratio
        11. Selector field used to split the data into two sets (labeled by the experts)

    Source: https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)

    :return:
    """
    url = (
        "https://archive.ics.uci.edu"
        + "/ml/machine-learning-databases/00225"
        + "/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    )

    response = requests.get(url)

    memfile = BytesIO()
    memfile.write(response.content)
    memfile.seek(0)

    df = pd.read_csv(memfile, header=None)
    target = df.iloc[:, -1] * 2 - 3

    data = df.iloc[:, 0:-1]
    data.columns = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag"]
    data["gender"] = (data["gender"] == "Female").astype("int")

    return data, target


if __name__ == '__main__':
    get_ilpd()
