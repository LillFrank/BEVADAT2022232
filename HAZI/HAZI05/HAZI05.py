

from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import pandas as pd


class KNNClassifier:

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio 

    
    @property
    def k_neighbors(self):
        return self.k
    


    def load_csv ( csv_path :str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        random.seed(42)
        dataset = pd.read_csv(csv_path)
        shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = shuffled.iloc[:,:-1],shuffled.iloc[:,-1] 
        return x,y
    

    def train_test_split(self, features: pd.DataFrame, labels: pd.DataFrame)-> None:
        test_size = int(len(features)* self.test_split_ratio)
        train_size= len(features)-test_size
        assert len(features) == test_size + train_size, "Size mismatch"

        self.x_train, self.y_train = features.iloc[:train_size,:] ,labels.iloc[:train_size]
        self.x_test, self.y_test = features.iloc[train_size:train_size+test_size,:] ,labels.iloc[train_size:train_size+test_size]


  
    
    def euclidean(self, element_of_x:pd.DataFrame)-> pd.DataFrame:
        return (self.x_train - element_of_x).pow(2).sum(axis=1).pow(1/2)
    

    def predict( self, x_test:pd.DataFrame) -> None:
        labels_pred = []
        for x_test_element in x_test:
            #tavolsagok meghatarozasa
            distances = self.euclidean( x_test_element)
            distances = pd.DataFrame(sorted(zip(distances, self.y_train)))

            #leggyakoribb label kiszedése:
            label_pred = mode(distances.iloc[:self.k,1],keepdims=False).mode
            labels_pred.append(label_pred)

        self.y_preds = pd.DataFrame(labels_pred, dtype=pd.int64)


    def accuracy(self)-> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test)*100
    

    def confusion_matrix(self)-> None:
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        return conf_matrix