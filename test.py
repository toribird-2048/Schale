import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


a = np.array([[0.1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4]])
print(sigmoid(a))