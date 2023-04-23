import numpy as np
from sklearn.model_selection import train_test_split


class LinearRegression:
       
    def __init__(self, epochs: int = 1000, lr: float = 1e-3): 
   
        self.epochs = epochs
        self.lr = lr
    

    def fit(self, X: np.array, y: np.array): # x: training data, y: target value
      #  self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.m = 0 #meredekség
        self.c = 0 #metszés x tengelyel
        n = float(len(X)) # Number of elements in X

 
        losses = []
        for i in range(self.epochs): 
            self.y_pred = self.m*X + self.c  # The current predicted value of Y

            residuals = self.y_pred - y
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2/n) * sum(X * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m + self.lr * D_m  # Update m
            self.c = self.c + self.lr * D_c  # Update c
            if i % 100 == 0:
                (np.mean(y-self.y_pred))


    def predict(self, X):
        pred = []
        for Xi in X:
            self.y_pred = self.m*Xi + self.c
            pred.append(self.y_pred)

        return pred


