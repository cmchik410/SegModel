from net.PSP import PSPnet
from losses.loss import pix_acc

def train(kwargs):
    img_shape = kwargs["dimensions"] + (kwargs["channels"], )
    
    m = PSPnet(img_shape, kwargs["n_classes"], kwargs["ppm_output_channel"], kwargs["pool_size"], kwargs["strides"])
    
    m.summary()
    
    m.compile(optimizer = "adam", loss = pix_acc)
    
    

    
# m = PSPnet(IMG_SHAPE, N_CLASSES, OUTPUT_CHANNEL, POOL_SIZE, STRIDES)
# m.build(INPUT_SHAPE)
# m.summary()