import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
   return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        #fit the training samples and labels

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        #list of labels
        return np.array(y_pred)
        #turn into array

    def _predict(self,x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
            #argsort will sort the distances until self.k (which is 3 default)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

