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
from sklearn.datasets import load_digits


class KMeansOnDigits():

    def __init__(self, n_clusters, random_state) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_digits(self):
        self.digits = load_digits()
    
    def predict(self)-> KMeans and  np.ndarray:
        kmean = KMeans(n_clusters=self.n_clusters,random_state=self.random_state)
        self.clusters = kmean.fit_predict(self.digits.data)

    def get_labels(self):
        result = np.empty(self.clusters.shape)

        for cl in self.digits.target_names:
            ms = (self.clusters == cl)
            subarray = self.digits.target[ms]
            mod = mode(subarray).mode.item()
            result[ms] = mod
        self.labels = result

    def calc_accuracy(self):
        self.accuracy = accuracy_score(self.digits.target,self.labels)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
       