from numpy import floor
from keras import Model
from keras.applications.resnet import ResNet50

from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import ReLU

from layers.maxarg import MaxArg

class PPM(Model):
    def __init__(self,
                 img_shape, 
                 output_channel, 
                 pooling_size, 
                 strides, 
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.H, self.W, self.C = img_shape
        
        self.ps1 = floor(self.H - ((pooling_size - 1) * strides)).astype(int)
        self.ps2 = floor(self.W - ((pooling_size - 1) * strides)).astype(int)
    
        
        self.avgPool1 = AveragePooling2D(pool_size = (self.ps1, self.ps2), strides = strides, padding = "same")
        self.Conv2D = Conv2D(output_channel, kernel_size = pooling_size, strides = strides, padding = "same")
        self.Bn1 = BatchNormalization()
        self.relu1 = ReLU()
        
        
    def call(self, inputs):
        x = self.avgPool1(inputs)
        x = self.Conv2D(x)
        x = self.Bn1(x)

        return self.relu1(x)



class PSPnet(Model):
    def __init__(self,
                 img_shape = (256, 256, 3), 
                 n_classes = 150, 
                 output_channel = 512, 
                 pooling_sizes = (1, 2, 3, 6), 
                 strides = 1, 
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.input_layer = Input(img_shape)
        
        self.res_layer =  ResNet50(include_top = False, weights = "imagenet", input_shape = img_shape)
        
        self.PPM_layers = []
        
        for ps in pooling_sizes:
            self.PPM_layers.append(PPM(img_shape, output_channel, ps, strides))
        
        self.concat = Concatenate()
        
        self.conv = Conv2D(filters = n_classes, kernel_size = 1, activation = "softmax", padding = "same")
        
        self.up= UpSampling2D(32)
        
        self.out = self.call(self.input_layer)

    def call(self, inputs):
        res = self.res_layer(inputs)
        
        x = res
        
        ppms = []
        for ppm_layer in self.PPM_layers:
            ppms.append(ppm_layer(x))
        
        for c in ppms:
            x = self.concat([x, c])
            
        x = self.conv(x)
        
        return self.up(x)