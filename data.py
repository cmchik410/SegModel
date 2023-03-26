import cv2
import numpy as np

def load_data(fpath, dims, normalize = True):
    res = []
    for f in fpath:
        temp_img = cv2.imread(f)
        temp_img = cv2.resize(temp_img, dims, interpolation = cv2.INTER_NEAREST)
        res.append(res)

    return np.array(res)