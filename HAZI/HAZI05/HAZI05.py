

from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import pandas as pd
import math


class KNNClassifier:

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio 

    
    @property
    def k_neighbors(self):
        return self.k
    

    @staticmethod
    def load_csv ( csv_path :str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = pd.read_csv(csv_path,delimiter=',')
        shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = shuffled.iloc[:,:-1],shuffled.iloc[:,-1] 
        return x,y
    

    def train_test_split(self, features: pd.DataFrame, labels: pd.DataFrame)-> None:
        test_size = int(len(features)* self.test_split_ratio)
        train_size= len(features)-test_size
        assert len(features) == test_size + train_size, "Size mismatch"

        self.x_train, self.y_train = features.iloc[:train_size,:].reset_index(drop=True) ,labels.iloc[:train_size].reset_index(drop=True)
        self.x_test, self.y_test = features.iloc[train_size:train_size+test_size,:].reset_index(drop=True) ,labels.iloc[train_size:train_size+test_size].reset_index(drop=True)


  
    
    def euclidean(self, element_of_x:pd.Series)-> pd.Series:
        return (self.x_train - element_of_x).pow(2).sum(axis=1).pow(1./2)
    

    def predict(self, x_test:pd.DataFrame):
        labels_pred = []
    
        for i,x_test_element in x_test.iterrows():
            #tavolsagok meghatarozasa
            distances = self.euclidean(x_test_element)
       
            distances = pd.DataFrame({'dis':distances, 'labs': self.y_train})
            distances.sort_values(by=['dis','lab'])
            #leggyakoribb label kiszedése:
            label_pred = mode(distances.iloc[:self.k,1],axis=0).mode
            labels_pred.append(label_pred[0])

        self.y_preds = pd.Series(labels_pred)
        




    def accuracy(self)-> float:
        true_positive = (self.y_test.reset_index(drop=True) == self.y_preds.reset_index(drop=True)).sum()
        return true_positive / len(self.y_test)*100
    

    def confusion_matrix(self)-> None:
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        return conf_matrix
    

    def best_k(self) -> Tuple[int, float]:
    
        best_ac= -math.inf
        best_k = -1
        actual_k = self.k
        for i in range(1,21):
            self.k = i
            self.predict(self.x_test)
            ac = self.accuracy()
            if best_ac < ac:
                best_ac = ac
                best_k = i

        self.k = actual_k
        return (round(best_ac,3), best_k)