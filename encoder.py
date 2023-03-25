import glob
from cv2 import INTER_NEAREST
import numpy as np
import cv2
from tqdm import tqdm

FPATH = "datasets/ADEChallengeData2016/annotations/**/*.png"
DIMS = (256, 256)
N_CLASSES = 151



def one_hot(targets, n_classes):
    res = np.eye(n_classes, dtype= np.int8)[np.array(targets, dtype=np.int8).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])


def encoder(fpath, dims, n_classes):
    filenames = glob.glob(fpath, recursive = True)
    
    filenames = filenames[500:2000]
    
    imgs = []
    for i in tqdm(filenames):
        temp = cv2.imread(i)
        temp = cv2.resize(temp, dims, interpolation = INTER_NEAREST)
        imgs.append(temp)
        
    imgs = np.array(imgs)
    
    imgs = np.max(imgs, axis = -1)
    
    res = one_hot(imgs, n_classes)
    
    return res

def main():
    res = encoder(FPATH, DIMS, N_CLASSES)
    
    print(res.shape)
    
main()