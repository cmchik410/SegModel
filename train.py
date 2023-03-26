import argparse
import yaml

import tensorflow as tf

from glob import glob
from tqdm import tqdm
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

from losses.loss import pix_acc
from net.PSP import build_PSPnet
from utils.data import load_data
from utils.encoder import one_hot



def train(**kwargs):
    X_train_path = glob(kwargs["train_path"], recursive = True)
    y_train_path = glob(kwargs["train_label_path"], recursive = True)

    img_shape = tuple(kwargs["dimensions"]) + (kwargs["channels"], )
    n_classes = kwargs["n_classes"]
    output_channels = kwargs["ppm_output_channels"]
    pooling_sizes = tuple(kwargs["pooling_sizes"])
    strides = kwargs["strides"]
    batch_size = kwargs["batch_size"]
    lr = kwargs["learning_rate"]
    epochs = kwargs["epochs"]

    steps = epochs / len(X_train_path)

    m = build_PSPnet(img_shape, n_classes, output_channels, pooling_sizes, strides)
    opt = Adam(learning_rate = lr)
    loss_fcn = BinaryCrossentropy()

    for ep in epochs:
        print("Epochs : %d / %d" %(ep, epochs))

        start = 0
        end = batch_size

        X_batch = load_data(X_train_path, img_shape[0:1]) / 255.
        y_true = load_data(y_train_path, img_shape[0:1])
        y_true = one_hot(y_true, n_classes)

        for stp in tqdm(range(steps)):
            with tf.GradientTape() as tape:
                y_pred = m(X_batch, training = True)

                loss_value = loss_fcn(y_true, y_pred)

        grads = tape.gradient(loss_value, m.trainable_weights)

        opt.apply_gradients(zip(grads, m.trainable_weights))

            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pyramid Scene Parsing Net 50 Training")

    parser.add_argument("--cfg", default = "configs/train_config.yaml", help = "path to train config file", type = str)

    args = parser.parse_args()

    with open(args.cfg, "r") as fp:
        kwargs = yaml.load(fp, Loader = yaml.FullLoader)

    train(**kwargs)