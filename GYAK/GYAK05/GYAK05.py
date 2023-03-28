# %%


import numpy as np
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
csv_path = "iris.csv"




class KNNClassifier:

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio 
    

    @property
    def k_neighbors(self):
        return self.k



    @staticmethod
    def load_csv ( csv_path :str) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        dataset = np.genfromtxt(csv_path, delimiter=',')
        np.random.shuffle(dataset)
        x,y = dataset[:,:-1],dataset[:,-1] 
        return x,y

    x,y = load_csv(csv_path)



    np.mean(x, axis=0),np.var(x,axis=0)
    np.nanmean(x,axis=0),np.nanvar(x,axis=0)
    x[np.isnan(x)] = 3.5
    (x>10.0).sum(), (x<0.0).sum()
    x[np.where(np.logical_or(x>10.0, x<0.0))]

    less_than = np.where(x<0.0)
    higher_than = np.where(x>10.0)
    less_than,higher_than

    y = np.delete(y,np.where(x<0.0)[0], axis=0)
    y = np.delete(y,np.where(x>10.0)[0], axis=0)

    x = np.delete(x,np.where(x<0.0)[0], axis=0)
    x = np.delete(x,np.where(x>10.0)[0], axis=0)



    
    def train_test_split(self, features: np.ndarray, labels: np.ndarray):
        test_size = int(len(features)* self.test_split_ratio)
        train_size= len(features)-test_size
        assert len(features)== test_size + train_size, "Size mismatch"

        self.x_train, self.y_train = features[:train_size,:] ,labels[:train_size]
        self.x_test, self.y_test = features[train_size:,:] ,labels[train_size:]

   



   
    def euclidean(self, element_of_x:np.ndarray)-> np.ndarray:
        return np.sqrt(np.sum((self.x_train - element_of_x)**2,axis=1))



    
    def predict( x_test:np.ndarray, self):
        labels_pred = []
        for x_test_element in x_test:
            #tavolsagok meghatarozasa
            distances = self.euclidean( x_test_element)
            distances = np.array(sorted(zip(distances, self.y_train)))

            #leggyakoribb label kiszedése:
            label_pred = mode(distances[:self.k,1],keepdims=False).mode
            labels_pred.append(label_pred)

        self.y_preds = np.array(labels_pred, dtype=np.int64)


    



    def accuracy(self)-> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test)*100



    def confusion_matrix(self)-> None:
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        sns.heatmap(conf_matrix,annot=True)
   


