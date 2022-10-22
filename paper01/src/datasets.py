from io import BytesIO
from typing import Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn import datasets
import requests_cache
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def url_request(url: str, name: str) -> BytesIO:
    session = requests_cache.CachedSession(
        cache_name=name,
        expire_after=timedelta(days=1)
    )
    response = session.get(url)

    memfile = BytesIO()
    memfile.write(response.content)
    memfile.seek(0)

    return memfile


def scaler(dataset: callable):
    def wrapper() -> Tuple[pd.DataFrame, pd.Series]:
        data, target = dataset()
        scaler_ = StandardScaler()
        data_scaled = pd.DataFrame(
            data=scaler_.fit_transform(data),
            columns=data.columns
        )
        return data_scaled, target
    return wrapper


@scaler
def get_linear(n_obs: int = 2048, n_feats: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
    data, target = datasets.make_blobs(
        n_samples=n_obs, n_features=n_feats,
        centers=[[-2, -2], [2, 2]],
        cluster_std=1.5,
    )
    target = (target == 1) * 2 - 1
    return pd.DataFrame(data), pd.Series(target)


@scaler
def get_blobs(n_obs: int = 2048, n_feats: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
    data, target = datasets.make_blobs(
        n_samples=n_obs, n_features=n_feats,
        centers=[
            [-2, -2], [2, 2],
            [-2, 2], [2, -2],
        ],
        cluster_std=1.2,
    )
    target = np.isin(target, [0, 1]) * 2 - 1
    return pd.DataFrame(data), pd.Series(target)


@scaler
def get_moons(n_obs: int = 2048) -> Tuple[pd.DataFrame, pd.Series]:
    data, target = datasets.make_moons(
        n_samples=n_obs, noise=0.2,
    )

    target = (target == 1) * 2 - 1
    data = data - [data[:, 0].mean(), data[:, 1].mean()]
    return pd.DataFrame(data), pd.Series(target)


@scaler
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

    df = pd.read_csv(url_request(url, "ilpd"), header=None)
    target = df.iloc[:, -1] * 2 - 3

    data = df.iloc[:, 0:-1]
    data.columns = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag"]
    data["gender"] = (data["gender"] == "Female").astype("int")

    return data, target


@scaler
def get_australian_credit() -> Tuple[pd.DataFrame, pd.Series]:
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

    df = pd.read_csv(url_request(url, "aus_credit"), header=None, sep=" ")
    target = df.iloc[:, -1] * 2 - 1

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
    data.columns = map(str, data.columns)

    return data, target


@scaler
def get_banknote() -> Tuple[pd.DataFrame, pd.Series]:
    """Data were extracted from images that were taken from genuine and forged
    banknote-like specimens. For digitization, an industrial camera usually used
    for print inspection was used. The final images have 400x 400 pixels. Due to
    the object lens and distance to the investigated object gray-scale pictures
    with a resolution of about 660 dpi were gained. Wavelet Transform tool were
    used to extract features from images.

    Attribute Information:
    1. variance of Wavelet Transformed image (continuous)
    2. skewness of Wavelet Transformed image (continuous)
    3. curtosis of Wavelet Transformed image (continuous)
    4. entropy of image (continuous)
    5. class (integer)

    Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/00267"
            + "/data_banknote_authentication.txt"
    )

    df = pd.read_csv(url_request(url, "banknote"), header=None, sep=",")
    target = df.iloc[:, -1] * 2 - 1
    data = df.iloc[:, :-1]
    data.columns = ["var", "skew", "kurt", "entropy"]

    return data, target


@scaler
def get_breast_cancer_coimbra() -> Tuple[pd.DataFrame, pd.Series]:
    """Data Set Information: There are 10 predictors, all quantitative, and a
    binary dependent variable, indicating the presence or absence of breast
    cancer. The predictors are anthropometric data and parameters which can be
    gathered in routine blood analysis. Prediction models based on these
    predictors, if accurate, can potentially be used as a biomarker of breast
    cancer.

    Attribute Information:
        Quantitative Attributes:
        Age (years)
        BMI (kg/m2)
        Glucose (mg/dL)
        Insulin (µU/mL)
        HOMA
        Leptin (ng/mL)
        Adiponectin (µg/mL)
        Resistin (ng/mL)
        MCP-1(pg/dL)

        Labels:
        1=Healthy controls
        2=Patients

    Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/00451"
            + "/dataR2.csv"
    )

    df = pd.read_csv(url_request(url, "breast_coimbra"), header=0)
    target = df.iloc[:, -1] * 2 - 3
    data = df.iloc[:, :-1]

    return data, target


@scaler
def get_breast_cancer_wisconsin() -> Tuple[pd.DataFrame, pd.Series]:
    """Data Set Information: Features are computed from a digitized image of a
    fine needle aspirate (FNA) of a breast mass. They describe characteristics
    of the cell nuclei present in the image. Separating plane described above
    was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett,
    "Decision Tree Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101,
    1992], a classification method which uses linear programming to construct a
    decision tree. Relevant features were selected using an exhaustive search in
    the space of 1-4 features and 1-3 separating planes.

    The actual linear program used to obtain the separating plane in the
    3-dimensional space is that described in: [K. P. Bennett and
    O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly
    Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

    #  Attribute                     Domain
    -- -----------------------------------------
    1. Sample code number            id number
    2. Clump Thickness               1 - 10
    3. Uniformity of Cell Size       1 - 10
    4. Uniformity of Cell Shape      1 - 10
    5. Marginal Adhesion             1 - 10
    6. Single Epithelial Cell Size   1 - 10
    7. Bare Nuclei                   1 - 10
    8. Bland Chromatin               1 - 10
    9. Normal Nucleoli               1 - 10
    10. Mitoses                      1 - 10
    11. Class:                       (2 for benign, 4 for malignant)

    Source: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/breast-cancer-wisconsin"
            + "/breast-cancer-wisconsin.data"
    )

    df = pd.read_csv(url_request(url, "breast_wisconsin"), header=None)
    df = (
        df
        .replace("?", np.nan)
        .astype("float")
        .dropna()
    )

    target = df.iloc[:, -1] - 3
    data = df.iloc[:, 1:-1]
    data.columns = [
        "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
        "marginal_adhesion", "single_epith_cell_size", "bare_nuclei",
        "bland_chromatin", "normal_nucleioli", "mitoses",
    ]

    return data, target


@scaler
def get_german_credit() -> Tuple[pd.DataFrame, pd.Series]:
    """Data Set Information: Two datasets are provided. the original dataset,
    in the form provided by Prof. Hofmann, contains categorical/symbolic
    attributes and is in the file "german.data".

    For algorithms that need numerical attributes, Strathclyde University
    produced the file "german.data-numeric". This file has been edited and
    several indicator variables added to make it suitable for algorithms which
    cannot cope with categorical variables. Several attributes that are ordered
    categorical (such as attribute 17) have been coded as integer. This was the
    form used by StatLog.

    It is worse to class a customer as good when they are bad, than it is to
    class a customer as bad when they are good.

    Attribute Information:

        Attribute 1: (qualitative)
        Status of existing checking account
        A11 : ... < 0 DM
        A12 : 0 <= ... < 200 DM
        A13 : ... >= 200 DM / salary assignments for at least 1 year
        A14 : no checking account

        Attribute 2: (numerical)
        Duration in month

        Attribute 3: (qualitative)
        Credit history
        A30 : no credits taken/ all credits paid back duly
        A31 : all credits at this bank paid back duly
        A32 : existing credits paid back duly till now
        A33 : delay in paying off in the past
        A34 : critical account/ other credits existing (not at this bank)

        Attribute 4: (qualitative)
        Purpose
        A40 : car (new)
        A41 : car (used)
        A42 : furniture/equipment
        A43 : radio/television
        A44 : domestic appliances
        A45 : repairs
        A46 : education
        A47 : (vacation - does not exist?)
        A48 : retraining
        A49 : business
        A410 : others

        Attribute 5: (numerical)
        Credit amount

        Attibute 6: (qualitative)
        Savings account/bonds
        A61 : ... < 100 DM
        A62 : 100 <= ... < 500 DM
        A63 : 500 <= ... < 1000 DM
        A64 : .. >= 1000 DM
        A65 : unknown/ no savings account

        Attribute 7: (qualitative)
        Present employment since
        A71 : unemployed
        A72 : ... < 1 year
        A73 : 1 <= ... < 4 years
        A74 : 4 <= ... < 7 years
        A75 : .. >= 7 years

        Attribute 8: (numerical)
        Installment rate in percentage of disposable income

        Attribute 9: (qualitative)
        Personal status and sex
        A91 : male : divorced/separated
        A92 : female : divorced/separated/married
        A93 : male : single
        A94 : male : married/widowed
        A95 : female : single

        Attribute 10: (qualitative)
        Other debtors / guarantors
        A101 : none
        A102 : co-applicant
        A103 : guarantor

        Attribute 11: (numerical)
        Present residence since

        Attribute 12: (qualitative)
        Property
        A121 : real estate
        A122 : if not A121 : building society savings agreement/ life insurance
        A123 : if not A121/A122 : car or other, not in attribute 6
        A124 : unknown / no property

        Attribute 13: (numerical)
        Age in years

        Attribute 14: (qualitative)
        Other installment plans
        A141 : bank
        A142 : stores
        A143 : none

        Attribute 15: (qualitative)
        Housing
        A151 : rent
        A152 : own
        A153 : for free

        Attribute 16: (numerical)
        Number of existing credits at this bank

        Attribute 17: (qualitative)
        Job
        A171 : unemployed/ unskilled - non-resident
        A172 : unskilled - resident
        A173 : skilled employee / official
        A174 : management/ self-employed/
        highly qualified employee/ officer

        Attribute 18: (numerical)
        Number of people being liable to provide maintenance for

        Attribute 19: (qualitative)
        Telephone
        A191 : none
        A192 : yes, registered under the customers name

        Attribute 20: (qualitative)
        foreign worker
        A201 : yes
        A202 : no

    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/statlog"
            + "/german/german.data"
    )

    df = pd.read_csv(url_request(url, "german_credit"), header=None, sep=" ")
    target = df.iloc[:, -1] * 2 - 3

    data = df.iloc[:, :-1]
    data.columns = [f"A{i}" for i in range(1, data.shape[1] + 1)]

    cats = data.select_dtypes("object")
    conts = data[[col for col in data.columns if col not in cats.columns]]

    encoder = OneHotEncoder(drop="first", sparse=False)
    data = pd.concat(
        [conts,
         pd.DataFrame(encoder.fit_transform(cats)).astype("int")],
        axis=1
    )
    data.columns = map(str, data.columns)

    return data, target


@scaler
def get_haberman_survival() -> Tuple[pd.DataFrame, pd.Series]:
    """Data Set Information: The dataset contains cases from a study that was
    conducted between 1958 and 1970 at the University of Chicago's Billings
    Hospital on the survival of patients who had undergone surgery for breast
    cancer.

    Attribute Information:
        1. Age of patient at time of operation (numerical)
        2. Patient's year of operation (year - 1900, numerical)
        3. Number of positive axillary nodes detected (numerical)
        4. Survival status (class attribute)
        -- 1 = the patient survived 5 years or longer
        -- 2 = the patient died within 5 year

    Source: https://archive.ics.uci.edu/ml/datasets/haberman%27s+survival

    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/haberman"
            + "/haberman.data"
    )

    df = pd.read_csv(url_request(url, "haberman_survival"), header=None, sep=",")
    target = df.iloc[:, -1] * 2 - 3
    data = df.iloc[:, :-1]
    data.columns = ["age", "operation_year", "n_nodes"]

    return data, target


@scaler
def get_sonar() -> Tuple[pd.DataFrame, pd.Series]:
    """Data Set Information: The file "sonar.mines" contains 111 patterns
    obtained by bouncing sonar signals off a metal cylinder at various angles
    and under various conditions. The file "sonar.rocks" contains 97 patterns
    obtained from rocks under similar conditions. The transmitted sonar signal
    is a frequency-modulated chirp, rising in frequency. The data set contains
    signals obtained from a variety of different aspect angles, spanning 90
    degrees for the cylinder and 180 degrees for the rock.

    Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number
    represents the energy within a particular frequency band, integrated over a
    certain period of time. The integration aperture for higher frequencies
    occur later in time, since these frequencies are transmitted later during
    the chirp.

    The label associated with each record contains the letter "R" if the object
    is a rock and "M" if it is a mine (metal cylinder). The numbers in the
    labels are in increasing order of aspect angle, but they do not encode the
    angle directly.

    Source: http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

    """

    url = (
            "http://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/undocumented/connectionist-bench"
            + "/sonar/sonar.all-data"
    )

    df = pd.read_csv(url_request(url, "sonar"), header=None, sep=",")
    target = (df.iloc[:, -1] == "M").astype("int") * 2 - 1
    data = df.iloc[:, :-1]
    data.columns = [f"A{i}" for i in range(data.shape[1])]

    return data, target


@scaler
def get_heart() -> Tuple[pd.DataFrame, pd.Series]:
    """This dataset is a heart disease database similar to a database already
    present in the repository (Heart Disease databases) but in a slightly
    different form

    Attribute Information:
    ------------------------
    -- 1. age
    -- 2. sex
    -- 3. chest pain type (4 values)
    -- 4. resting blood pressure
    -- 5. serum cholesterol in mg/dl
    -- 6. fasting blood sugar > 120 mg/dl
    -- 7. resting electrocardiographic results (values 0,1,2)
    -- 8. maximum heart rate achieved
    -- 9. exercise induced angina
    -- 10. oldpeak = ST depression induced by exercise relative to rest
    -- 11. the slope of the peak exercise ST segment
    -- 12. number of major vessels (0-3) colored by flourosopy
    -- 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

    Variable to be predicted
    ------------------------
    Absence (1) or presence (2) of heart disease

    Source: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29

    """

    url = (
            "https://archive.ics.uci.edu"
            + "/ml/machine-learning-databases/statlog"
            + "/heart/heart.dat"
    )

    df = pd.read_csv(url_request(url, "heart"), header=None, sep=" ")
    target = df.iloc[:, -1] * 2 - 3
    data = df.iloc[:, :-1]
    data.columns = [
        "age", "sex", "chest_pain_type", "rest_blood_pressure", "serum_cholesterol",
        "fasting_blood_sugar", "resting_ekg", "heart_rate", "exercise_induced_angina",
        "oldpeak", "slope_peak", "n_major_vessels", "thal",
    ]

    cats = data[[
        "sex", "chest_pain_type", "fasting_blood_sugar", "resting_ekg",
        "exercise_induced_angina", "thal",
    ]]
    conts = data[[col for col in data.columns if col not in cats.columns]]

    encoder = OneHotEncoder(drop="first", sparse=False)
    data = pd.concat(
        [conts,
         pd.DataFrame(encoder.fit_transform(cats)).astype("int")],
        axis=1
    )
    data.columns = map(str, data.columns)

    return data, target


def alldts() -> dict:
    return {
        "synth_linear": get_linear(),
        "synth_blobs": get_blobs(),
        "synth_moons": get_moons(),
        "ilpd": get_ilpd(),
        "cred_aus": get_australian_credit(),
        "cred_ger": get_german_credit(),
        "banknote": get_banknote(),
        "breast_coimbra": get_breast_cancer_coimbra(),
        "breast_wiscons": get_breast_cancer_wisconsin(),
        "haberman_surv": get_haberman_survival(),
        "sonar": get_sonar(),
        "heart": get_heart(),
    }


if __name__ == '__main__':
    # TEST FUNCTIONS EXECUTION

    import sys
    from inspect import getmembers, isfunction

    funcs = getmembers(sys.modules[__name__], isfunction)
    for func in filter(lambda x: x[0].startswith("get_"), funcs):
        print(func[0], end=" ")
        data_, target_ = func[1]()
        assert sorted(list(target_.value_counts().index)) == [-1, 1]
        print("OK")
