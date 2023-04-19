from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout


def build_unet(img_shape, n_classes):
    inputs = Input(shape = img_shape)
    
    cc = {}
    
    x = inputs
    
    for i in range(4):
        f = 64 * (2 ** i)
        x = Conv2D(f , 3, activation = "relu", kernel_initializer = "he_normal", padding = "same")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(f , 3, activation = "relu", kernel_initializer = "he_normal", padding = "same")(x)
        cc['c' + str(i + 1)] = x
        x = MaxPooling2D(2)(x)
    
    x = Conv2D(f * 2 , 3, activation = "relu", kernel_initializer = "he_normal", padding = "same")(x)
    x = Dropout(0.3)(x)
    x = Conv2D(f * 2, 3, activation = "relu", kernel_initializer = "he_normal", padding = "same")(x)
    
    #Expand
    for i in range(3, -1, -1):
        f = 16 * (2 ** i)
        x = Conv2DTranspose(f, 2, 2, padding = "same")(x)
        x = concatenate([x, cc['c' + str(i + 1)]])
        x = Conv2D(f, 3, activation = "relu", kernel_initializer = "he_normal", padding = "same")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(f, 3, activation = "relu", kernel_initializer = "he_normal", padding = "same")(x)
    
    x = Conv2D(n_classes, 1, activation = 'sigmoid')(x)
    
    model = Model(inputs = inputs, outputs = x)
    
    return model