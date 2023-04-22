import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt





def load_iris_data() -> sklearn.utils.Bunch:
    return load_iris()


def check_data(iris) -> pd.DataFrame:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df.head(5)

def linear_train_data(iris) -> np.ndarray:
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    X = df[['sepal width (cm)','petal length (cm)','petal width (cm)']].values
    y = df['sepal length (cm)'].values
    return X,y

def logistic_train_data(iris) -> np.ndarray:
     df = pd.DataFrame(iris.data,columns=iris.feature_names)
     df['target'] = iris.target
     df.drop(df.loc[df['target'] == 2].index, inplace = True)
     y= df['target']
     X = df[['sepal length (cm)','sepal width (cm)', 'petal length (cm)','petal width (cm)']].values
     return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    LinearRegression().__init__()
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model

def train_logistic_regression(X_train, y_train):
    LogisticRegression().__init__()
    model = LogisticRegression()
    model.fit(X_train,y_train)
    return model

def predict(model , X_test):
    return model.predict(X_test)

def plot_actual_vs_predicted(y_test,y_pred):
    plt.scatter(y_test,y_pred)
    plt.ylabel="Predict"
    plt.xlabel="Actual"
    
    
def evaluate_model(y_test,y_pred):
    return mean_squared_error(y_test,y_pred)