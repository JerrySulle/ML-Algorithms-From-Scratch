import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init parameters (initial weight)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            #print("y_pred: ",y_predicted)
            dw =  ( 2 * np.dot(X.T, (y_predicted - y))) / n_samples
            #print("dw: ", dw)
            #X transposed
            db = ( 2 * np.sum(y_predicted - y)) / n_samples
            #print("db: ", db)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted