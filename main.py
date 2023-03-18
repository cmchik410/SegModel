from net.PSP import PSPnet

INPUT_SHAPE = (None, 256, 256, 3)
IMG_SHAPE = (256, 256, 3)
N_CLASSES = 21
OUTPUT_CHANNEL = 512
STRIDES = 1
POOL_SIZE = (1, 2, 3, 6)

def main():
    
    m = PSPnet(IMG_SHAPE, N_CLASSES, OUTPUT_CHANNEL, POOL_SIZE, STRIDES)
    m.build(INPUT_SHAPE)
    m.summary()

main()