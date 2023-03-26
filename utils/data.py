import cv2
import numpy as np
from glob import glob

def data_shuffle(X_path, y_path):
    X_path = np.array(sorted(glob(X_path, recursive = True)))
    y_path = np.array(sorted(glob(y_path, recursive = True)))

    shuffler = np.random.permutation(X_path.shape[0])

    return X_path[shuffler], y_path[shuffler]


def load_data(fpath, dims):
    res = []
    for f in fpath:
        temp_img = cv2.imread(f)
        temp_img = cv2.resize(temp_img, dims, interpolation = cv2.INTER_NEAREST)
        res.append(temp_img)

    return np.array(res)