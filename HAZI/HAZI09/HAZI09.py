# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import sklearn.datasets as datasets


class KMeansOnDigits():

    def __init__(self, n_clusters, random_state) -> None:
        self.nn_clusters = n_clusters
        self.random_state = random_state

    def load_digits(self):
        self.digits =  datasets.load_digits()
    
    def predict(self)-> KMeans and  np.ndarray:
        kmean = KMeans(n_clusters=self.n_clusters,random_state=self.random_state)
        self.clusters = kmean.fit_predict(self.digits.data, self.digits.target)

    def get_labels(self):
        pass

    def calc_accuracy(self):
        pass

    def confusion_matrix(self):
        pass
       