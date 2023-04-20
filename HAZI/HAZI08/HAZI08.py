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

def linear_train_data(iris) -> np.ndarray:
    X = iris['sepal width (cm)'].values
    y = iris['sepal length (cm)'].values
    return X,y

def logistic_train_data(iris) -> np.ndarray:
     df = pd.DataFrame(iris.data,columns=iris.feature_names)
     df['target'] = iris.target
     df.drop(df.loc[df['target'] == 2].index, inplace = True)
     y= df['target']
     train = df[['sepal length (cm)','sepal width (cm)']].to_numpy()
     X= train
     return X,y