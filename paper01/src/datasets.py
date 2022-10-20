from io import BytesIO
from typing import Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn import datasets
import requests_cache
from sklearn.preprocessing import OneHotEncoder


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


def get_ilpd() -> Tuple[pd.DataFrame, pd.Series]:
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
    """
    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/00225"
            + "/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    )

    session = requests_cache.CachedSession(cache_name="ilpd", expire_after=timedelta(days=1))
    response = session.get(url)

    memfile = BytesIO()
    memfile.write(response.content)
    memfile.seek(0)

    df = pd.read_csv(memfile, header=None)
    target = df.iloc[:, -1] * 2 - 3

    data = df.iloc[:, 0:-1]
    data.columns = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag"]
    data["gender"] = (data["gender"] == "Female").astype("int")

    return data, target


def get_statlog() -> Tuple[pd.DataFrame, pd.Series]:
    """Statlog (Australian Credit Approval) Data Set

    Data Set Information: This file concerns credit card applications. All
    attribute names and values have been changed to meaningless symbols to
    protect confidentiality of the data.

    This dataset is interesting because there is a good mix of
    attributes -- continuous, nominal with small numbers of values, and nominal
    with larger numbers of values. There are also a few missing values.

    Attribute Information: There are 6 numerical and 8 categorical attributes.
    The labels have been changed for the convenience of the statistical
    algorithms. For example, attribute 4 originally had 3 labels p,g,gg and
    these have been changed to labels 1,2,3.

    A1: 0,1 CATEGORICAL (formerly: a,b)
    A2: continuous.
    A3: continuous.
    A4: 1,2,3 CATEGORICAL (formerly: p,g,gg)
    A5: 1,2,3,4,5,6,7,8,9,10,11,12,13,14 CATEGORICAL (formerly: ff,d,i,k,j,aa,m,c,w,e,q,r,cc,x)
    A6: 1,2,3,4,5,6,7,8,9 CATEGORICAL (formerly: ff,dd,j,bb,v,n,o,h,z)
    A7: continuous.
    A8: 1, 0 CATEGORICAL (formerly: t, f)
    A9: 1, 0 CATEGORICAL (formerly: t, f)
    A10: continuous.
    A11: 1, 0 CATEGORICAL (formerly t, f)
    A12: 1, 2, 3 CATEGORICAL (formerly: s, g, p)
    A13: continuous.
    A14: continuous.
    A15: 1,2 class attribute (formerly: +,-)

    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/statlog"
            + "/australian/australian.dat"
    )

    session = requests_cache.CachedSession(cache_name="statlog", expire_after=timedelta(days=1))
    response = session.get(url)

    memfile = BytesIO()
    memfile.write(response.content)
    memfile.seek(0)

    df = pd.read_csv(memfile, header=None, sep=" ")
    target = df.iloc[:, -1] * 2 - 3

    data = df.iloc[:, :-1]
    data.columns = [f"A{i}" for i in range(1, data.shape[1] + 1)]
    data = data.astype({
        "A1": "category",
        "A2": "float",
        "A3": "float",
        "A4": "category",
        "A5": "category",
        "A6": "category",
        "A7": "float",
        "A8": "category",
        "A9": "category",
        "A10": "int",
        "A11": "category",
        "A12": "category",
        "A13": "int",
        "A14": "int",
    })

    cats = data.select_dtypes("category")
    conts = data[[col for col in data.columns if col not in cats.columns]]

    encoder = OneHotEncoder(drop="first", sparse=False)
    data = pd.concat(
        [conts,
         pd.DataFrame(encoder.fit_transform(cats)).astype("int")],
        axis=1
    )

    return data, target
