import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class Node():
    def __init__(self, feature_index = None, threshold = None, left=None, right=None, info_gain=None, value = None) -> None:
        #leaf
        self.value = value

        #dis
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        self.root = None
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        

    def build_tree(self, dataset, curr_depth=0) -> Node:

        X, Y = dataset[:,:-1], dataset[:,-1] #x: all columns except the last one, y:the last column
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features) # best_split: dict
            if best_split["info_gain"] > 0: 
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    

    def get_best_split(self, dataset, num_samples, num_features) -> dict:
        ''' function to find the best split '''
        
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values) #possible_thersholds: indices of the unique values
            for threshold in possible_thresholds: 
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold) # dataset_left,dataset_right : np.array
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y = dataset[:, -1]
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]

                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"): 
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        

        
col_name = ['stop_sequence', 'from_id', 'to_id', 'status', 'line','type','day','delay']
dataN = pd.read_csv("data/NJ.csv", skiprows=1, header=None, names=col_name)


X = dataN.iloc[:,:-1].values
Y = dataN.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split=5.5, max_depth=5.5)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))

#c = [7.3, 7.5, 7.65, 7.8, 7.9]
#scores = []
#i=0
#for choice in c:
#    classifier = DecisionTreeClassifier(min_samples_split=c[i], max_depth=c[i])
#    classifier.fit(X_train, Y_train)
#    Y_pred = classifier.predict(X_test)
#    scores.append(accuracy_score(Y_test, Y_pred))
#    i=i+1

#scores


#4.feladat
'''
Amikor neki álltam a házinak először az órán leírt kódokat próbáltam értelmezni, 
mert sajnos bent az órán semmit sem tudtam megértetni (olyan gyorsan haladtuk). 
Először a default értékekkel (3,3) próbáltam ki a fit-elést, aminek az accuracy értéke 76% lett.
Ezután előbb az egész számokkal próbálkoztam:
(1,1)= 78%, (2,2)=78.5%, (4,4)=78.6%  (5,5)=78.7%, (6,6) = error,(7,7)=error, a 6 efölötti egész számokra error-t dobott ki.
Ezután megcéloztam az 5 környéki számokat mert ott éretem el eddig a legjobb eredményt.
(5.2,5.2)= 78.7%,(5.5,5.5)= 78.5% (5.9,5.9) = 78.75%
A 6 fölötti értékekre még minidg error-t dobott ki, illetve sehogy se jött ki 78.7%-nál jobb accuracy. 
Így kipróbáltam a moddle-ra feltöltött NJ_60k.csv-vel is és 
így (7.9,7.9) értékekkel elértem a 79.18%-ot de ennél jobbat sajnos nem sikerült elérnem és a 8 fölötti értékekre errort dobott ki.
Azt sajnos nem sikerült kitalálnom, hogy az én njcleaner-em által létrehozott csv miben volt más mint a feltöltött.
'''
        