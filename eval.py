import argparse
import yaml
import json
import cv2

from keras.models import load_model
from utils.encoder import decoder


def visualize_result(y_pred, color_dict_path, channels):
    color_dict = {}
    temp_dict = {}
    with open(color_dict_path, 'r') as fp:
        temp_dict = json.load(fp)
        
    for k, v in temp_dict.items():
        color_dict[int(k)] = v

    seg_img = decoder(y_pred, color_dict, channels)
    
    #cv2.imshow("Seg", seg_img)

def eval(**kwargs):
    img_path = kwargs["img_path"]
    model_path = kwargs["model_path"]
    color_dict_path = kwargs["color_dict"]
    img_shape = tuple(kwargs["dimensions"]) + (kwargs["channels"], )

    img = cv2.imread(img_path)
    img = cv2.resize(img, img_shape[0:2], interpolation = cv2.INTER_NEAREST)
    img = img.reshape((1,) + img_shape)

    m = load_model(model_path)
    m.summary()
    
    y_pred = m(img)
    
    visualize_result(y_pred[0], color_dict_path, img_shape[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Viewer")

    parser.add_argument("--cfg", default = "configs/eval_config.yaml", help = "path to eval config file", type = str)

    args = parser.parse_args()

    with open(args.cfg, "r") as fp:
        kwargs = yaml.load(fp, Loader = yaml.FullLoader)

    eval(**kwargs)