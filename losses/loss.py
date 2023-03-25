import numpy as np

def pix_acc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis = -1)
    valid = (y_true >= 0)
    acc_sum = np.sum((y_pred == y_true))
    pixel_sum = np.sum(valid)
    acc = float(acc_sum) / (float(pixel_sum) + 1e-10)
    return acc
