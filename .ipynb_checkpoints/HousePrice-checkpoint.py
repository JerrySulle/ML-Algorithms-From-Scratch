import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as plt

data  = pd.read_csv("USA_Housing.csv")

y = data["Price"]
X = data.drop(["Price","Address"],axis=1)
#n_samples, n_features = data.shape
#n_features -= 1
#X = data[:,0:n_features]
#y = data[:,n_features]

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

from LinearRegression import LinearRegression

LR = LinearRegression(lr=0.01, n_iters=1000)

LR.fit(X_train, y_train)
predictions = LR.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

mse_value = mse(y_test, predictions)
print(mse_value)

Y_pred = LR.predict(X_test)

print("Predicted values ", np.round( Y_pred[:3], 2 ))

print("Real values      ", y_test[:3])
