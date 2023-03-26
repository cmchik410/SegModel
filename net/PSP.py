
from numpy import floor
from keras import Model
from keras.applications.resnet import ResNet50

from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.layers import ReLU

def PPM(x, img_shape, output_channels, kernel_size, strides):
    H, W, C = img_shape

    pool_size = floor(H - ((kernel_size - 1) * strides)).astype(int)

    x = AveragePooling2D(pool_size = (pool_size, pool_size), strides = strides, padding = "same")(x)
    x = Conv2D(output_channels, kernel_size = kernel_size, strides = strides, padding = "same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x
    

def build_PSPnet(img_shape, n_classes, output_channels, pooling_sizes, strides):
    inputs = Input(img_shape)

    x = ResNet50(include_top = False, weights = "imagenet", input_shape = img_shape)(inputs)

    concatenation_layers = []
    for ps in pooling_sizes:
        concatenation_layers.append(PPM(x, img_shape, output_channels, ps, strides))
    
    for cl in concatenation_layers:
        x = concatenate([x, cl])
    
    x = Conv2D(filters = n_classes, kernel_size = 1, activation = "softmax", padding = "same")(x)

    x = UpSampling2D((32, 32))(x)

    model = Model(inputs = inputs, outputs = x)

    return model



