import argparse
import yaml
import json

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from net.PSP import build_PSPnet
from callbacks.callbacks import Reduce_LR, get_best_model
from utils.data import data_shuffle
from utils.training import train_step, val_step


def main(**kwargs):
    # Initializing Parameters

    X_train_path, y_train_path = data_shuffle(kwargs["train_path"], kwargs["train_label_path"])
    X_val_path, y_val_path = data_shuffle(kwargs["val_path"], kwargs["val_label_path"])

    # X_train_path = X_train_path[0:48]
    # y_train_path = y_train_path[0:48]
    # X_val_path = X_val_path[0:32]
    # y_val_path = y_val_path[0:32]

    img_shape = tuple(kwargs["dimensions"]) + (kwargs["channels"], )
    n_classes = kwargs["n_classes"]
    output_channels = kwargs["ppm_output_channels"]
    pooling_sizes = tuple(kwargs["pooling_sizes"])
    strides = kwargs["strides"]
    batch_size = kwargs["batch_size"]
    lr = kwargs["learning_rate"]
    epochs = kwargs["epochs"]
    save_model_path = kwargs["save_model_path"]

    # Preparing Training Model
    m = build_PSPnet(img_shape, n_classes, output_channels, pooling_sizes, strides)
    opt = SGD(learning_rate = lr)
    loss_fcn = CategoricalCrossentropy()
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()

    m.summary()
    
    history = {"train_loss" : [], "train_acc" : [], "val_loss" : [], "val_acc" : []}
    
    best_model = None

    # Start Training Loop
    for ep in range(1, epochs + 1):
        print("Epochs : %d / %d" %(ep, epochs))
        
        m, train_loss, train_acc = train_step(m, X_train_path, y_train_path, img_shape, n_classes, batch_size, opt, loss_fcn, train_acc_metric)

        # Validation
        
        m, val_loss, val_acc = val_step(m, X_val_path, y_val_path, img_shape, n_classes, batch_size, loss_fcn, val_acc_metric)
        
        print()
    
        history["train_loss"].append(train_loss.numpy().tolist())
        history["train_acc"].append(train_acc.numpy().tolist())
        history["val_loss"].append(val_loss.numpy().tolist())
        history["val_acc"].append(val_acc.numpy().tolist())
        
        Reduce_LR(opt, "val_loss", history, decay = 0.9, patience = 5, minimum = 1e-4, verbose = 1)
        
        if get_best_model("val_loss", history, patience = 5, verbose = 1):
            m = best_model
        else:
            best_model = m
    
    with open("history.txt", "w") as fp:
        json.dump(history, fp)    

    m.save(save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pyramid Scene Parsing Net 50 Training")

    parser.add_argument("--cfg", default = "configs/train_config.yaml", help = "path to train config file", type = str)

    args = parser.parse_args()

    with open(args.cfg, "r") as fp:
        kwargs = yaml.load(fp, Loader = yaml.FullLoader)

    main(**kwargs)