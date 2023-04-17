import cv2
import numpy as np
from glob import glob

def data_shuffle(X_path, y_path, ratio = 1):
    X_path = np.array(sorted(glob(X_path, recursive = True)))
    y_path = np.array(sorted(glob(y_path, recursive = True)))
    
    N = int(X_path.shape[0] * ratio)

    shuffler = np.random.permutation(X_path.shape[0])

    return X_path[shuffler][0:N], y_path[shuffler][0:N]

def load_data(fpath, dims):
    res = []
    for f in fpath:
        temp_img = cv2.imread(f)
        temp_img = cv2.resize(temp_img, dims, interpolation = cv2.INTER_NEAREST)
        res.append(temp_img)

    return np.array(res)