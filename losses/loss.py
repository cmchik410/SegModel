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


def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)  

def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)