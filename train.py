import argparse
import yaml

from time import sleep
from tensorflow import GradientTape 
from keras.utils import Progbar
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy

from losses.loss import NLL, pix_acc
from net.PSP import build_PSPnet
from utils.data import load_data, data_shuffle
from utils.encoder import one_hot

def train(**kwargs):
    # Initializing Parameters

    X_train_path, y_train_path = data_shuffle(kwargs["train_path"], kwargs["train_label_path"])
    X_val_path, y_val_path = data_shuffle(kwargs["val_path"], kwargs["val_label_path"])

    # X_train_path = X_train_path[0:2048]
    # y_train_path = y_train_path[0:2048]

    img_shape = tuple(kwargs["dimensions"]) + (kwargs["channels"], )
    n_classes = kwargs["n_classes"]
    output_channels = kwargs["ppm_output_channels"]
    pooling_sizes = tuple(kwargs["pooling_sizes"])
    strides = kwargs["strides"]
    batch_size = kwargs["batch_size"]
    lr = kwargs["learning_rate"]
    epochs = kwargs["epochs"]
    save_model_path = kwargs["save_model_path"]

    total_examples = len(X_train_path)
    steps = total_examples // batch_size

    # Preparing Training Model
    m = build_PSPnet(img_shape, n_classes, output_channels, pooling_sizes, strides)
    opt = SGD(learning_rate = lr)
    loss_fcn = CategoricalCrossentropy()
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()
    metrics_names = ['loss','acc'] 

    m.summary()

    # Start Training Loop
    for ep in range(1, epochs + 1):
        print("Epochs : %d / %d" %(ep, epochs))

        start = 0
        end = batch_size

        pb = Progbar(total_examples, stateful_metrics = metrics_names)

        for stp in range(steps):
            # Training 

            X_batch = load_data(X_train_path[start:end], img_shape[0:2])
            y_true = load_data(y_train_path[start:end], img_shape[0:2])
            y_true = one_hot(y_true, n_classes)

            with GradientTape() as tape:
                y_pred = m(X_batch, training = True)

                loss_value = loss_fcn(y_true, y_pred)

                train_acc_metric.update_state(y_true, y_pred)

                #train_acc = train_acc_metric(y_true, y_pred)

            grads = tape.gradient(loss_value, m.trainable_weights)

            opt.apply_gradients(zip(grads, m.trainable_weights))

            train_acc = train_acc_metric.result()

            train_acc_metric.reset_states()

            # # Validation

            # X_val_batch = load_data(X_val_path[0:batch_size], img_shape[0:2]) / 255.
            # y_val_true = load_data(y_val_path[0:batch_size], img_shape[0:2])
            # y_val_true = one_hot(y_val_true, n_classes)

            # y_val_pred = m(X_val_batch, training = False)

            # val_acc_metric.update_state(y_val_true, y_val_pred)

            # val_acc = val_acc_metric.result()
            # val_acc_metric.reset_states()


            # Update Progress Bar
            sleep(0.3)
        
            values=[('loss', loss_value), ('acc', train_acc)]
        
            pb.add(batch_size, values = values)

            start += batch_size
            end += batch_size

    m.save(save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pyramid Scene Parsing Net 50 Training")

    parser.add_argument("--cfg", default = "configs/train_config.yaml", help = "path to train config file", type = str)

    args = parser.parse_args()

    with open(args.cfg, "r") as fp:
        kwargs = yaml.load(fp, Loader = yaml.FullLoader)

    train(**kwargs)