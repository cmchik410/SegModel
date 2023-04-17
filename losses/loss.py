import numpy as np
from keras import backend as K
import tensorflow as tf

# Negative Log Loss
def NLL(y_true, y_pred):
    temp = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    temp = temp.reshape(-1)

    return tf.math.log(np.sum(temp, axis = -1))

def pix_acc(y_true, y_pred):
    total_pix = y_true.shape[0]
    acc_sum = np.sum((y_pred == y_true))
    acc = float(acc_sum) / (float(total_pix) + 1e-10)
    return acc
