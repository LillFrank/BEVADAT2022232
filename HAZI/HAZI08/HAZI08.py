import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error





def load_iris_data() -> sklearn.utils.Bunch:
    return load_iris()


def check_data(iris) -> pd.DataFrame:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df.head(5)