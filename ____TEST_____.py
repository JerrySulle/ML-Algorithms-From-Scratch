import numpy as np

# init mean, var, priors
_mean = np.zeros((2,3), dtype=np.float64)
_var = np.zeros((4, 5), dtype=np.float64)
_priors = np.zeros((2), dtype=np.float64)

x = np.array([[1,2,3], [4,5,6]])

print(_mean.shape)
print(x[0, :])

print(_var)

print(_priors)