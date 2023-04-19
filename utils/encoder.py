import numpy as np

def one_hot(targets, n_classes):
    targets = np.max(targets, axis = -1)
    
    res = np.eye(n_classes, dtype = np.int8)[np.array(targets, dtype=np.int8).reshape(-1)]

    return res.reshape(list(targets.shape)+[n_classes])



def decoder(targets, color_dict, channels = 3):
    img_shape = targets.shape[:2] + (channels,)
    single_layer = np.argmax(targets, axis = -1)
    output = np.zeros(img_shape)

    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]

    return np.uint8(output)