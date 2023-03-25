import tensorflow as tf
import numpy as np
import glob
import cv2

from net.PSP import PSPnet
from losses.loss import pix_acc
from data import load_data
from encoder import one_hot
from keras.metrics import Mean, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam


class trainAPI(object):
    def __init__(self, **kwargs):
        self.X_train = glob.glob(kwargs["train_path"])
        self.y_train = glob.glob(kwargs["train_label_path"])
        self.X_val = glob.glob(kwargs["val_path"])
        self.y_val = glob.glob(kwargs["val_label_path"])
        self.total_train = len(self.X_train)
        self.total_val = len(self.X_val)
        
        self.dims = kwargs["dimensions"]
        self.ch = kwargs["channels"]
        self.img_shape = self.dims + (self.ch, )
        self.n_classes = kwargs["n_classes"]
        self.out_ch = kwargs["ppm_output_channel"]
        self.pool_size = kwargs["pool_size"]
        self.strides = kwargs["strides"]
        
        self.batch_size = kwargs["batch_size"]
        self.n_epochs = kwargs["epochs"]
        self.n_steps = self.total_train // self.batch_size
        self.lr = kwargs["learning_rate"]
        
        self.save_model_path = kwargs["model_path"]
    
    def print_status_bar(self, iteration, total, loss, metrics = None):
        metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) 
                                        for m in [loss] + (metrics or [])])
        end = " " if iteration < total else "\n"
        print("\r{}/{} - ".foramt(iteration, total) + metrics, end = end)
    
    def run(self):
        m = PSPnet(self.img_shape, self.n_classes, self.out_ch, self.pool_size, self.strides)
        opt = Adam(learning_rate = self.lr)
        loss_fn = pix_acc
        mean_loss = Mean()
        metrics = MeanAbsoluteError()

        random_idx = np.random.permutation(np.arange(self.total_train))
        
        start = 0
        end = self.batch_size

        for epoch in range(1, self.n_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            X_batch = load_data(self.X_train[random_idx[start:end]])
            y_batch = load_data(self.y_train[random_idx[start:end]])
            y_batch = one_hot(y_batch, self.n_classes)
                            
            for step in range(1, self.n_steps + 1):
                with tf.GradientTape() as tape:
                    y_pred = m(X_batch, training = True)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + m.losses)
                    
                gradients = tape.gradient(loss, m.trainable_variables)
                opt.apply_gradients(zip(gradients, m.trainable_variables))

            start += self.batch_size
            end += self.batch_size

            self.print_status_bar(step * self.batch_size, len(self.total_train), mean_loss, metrics)

            for metric in [mean_loss] + metrics:
                metric.reset_states()

        
    
# m = PSPnet(IMG_SHAPE, N_CLASSES, OUTPUT_CHANNEL, POOL_SIZE, STRIDES)
# m.build(INPUT_SHAPE)
# m.summary()