import numpy as np

def one_hot(targets, n_classes):
    targets = np.max(targets, axis = -1)

    res = np.eye(n_classes, dtype = np.int8)[np.array(targets, dtype=np.int8).reshape(-1)]

    return res.reshape(list(targets.shape)+[n_classes])
