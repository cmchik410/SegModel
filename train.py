import tensorflow as tf
import glob
import cv2
from net.PSP import PSPnet
from tensorflow.keras.optimizers import Adam
from losses.loss import pix_acc

class trainAPI(object):
    def __init__(self, **kwargs):
        self.X_train = glob.glob(kwargs["train_path"])
        self.y_train = glob.glob(kwargs["train_label_path"])
        self.X_val = glob.glob(kwargs["val_path"])
        self.y_val = glob.glob(kwargs["val_label_path"])
        
        self.img_shape = kwargs["dimensions"] + (kwargs["channels"], )
        self.n_classes = kwargs["n_classes"]
        self.out_ch = kwargs["ppm_output_channel"]
        self.pool_size = kwargs["pool_size"]
        self.strides = kwargs["strides"]
        
        self.batch_size = kwargs["batch_size"]
        self.n_epochs = kwargs["epochs"]
        self.n_steps = len(self.X_train) // self.batch_size
        self.lr = kwargs["learning_rate"]
        
        self.save_model_path = kwargs["model_path"]
         
    
    def run(self):
        m = PSPnet(self.img_shape, self.n_classes, self.out_ch, self.pool_size, self.strides)
        opt = Adam(lr = self.lr)
        loss_fn = pix_acc
        
        for epoch in range(self.n_epochs):
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            
            
            
            with tf.GradientTape() as tape:
                y_pred = m(X_batch, training = True)
                main_loss = tf.reduce_mean(loss_fcn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + m.losses)
                
            gradients = tape.gradient(loss, m.trainable_variables)
            opt.apply_gradients(zip(gradients, m.trainable_variables))
    
# m = PSPnet(IMG_SHAPE, N_CLASSES, OUTPUT_CHANNEL, POOL_SIZE, STRIDES)
# m.build(INPUT_SHAPE)
# m.summary()