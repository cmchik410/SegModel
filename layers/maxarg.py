from tensorflow import argmax
from keras.layers import Layer


class MaxArg(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return argmax(x, axis = 3)

