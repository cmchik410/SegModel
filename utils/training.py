from time import sleep

from tensorflow.keras.utils import Progbar
from tensorflow import GradientTape
 
from utils.data import load_data
from utils.encoder import one_hot

def train_step(model, X_path, y_path, img_shape, n_classes, batch_size, opt, loss_fcn, metric):
    n_samples = len(X_path)
    n_steps = n_samples // batch_size
    
    pb = Progbar(n_samples)
    
    start = 0
    end = batch_size
    
    avg_loss = 0.0
    avg_acc = 0.0

    for stp in range(1, n_steps + 1):
        # Training 
        X_batch = load_data(X_path[start:end], img_shape[0:2])
        y_true = load_data(y_path[start:end], img_shape[0:2])
        y_true = one_hot(y_true, n_classes)

        with GradientTape() as tape:
            y_pred = model(X_batch, training = True)

            loss = loss_fcn(y_true, y_pred)

            metric.update_state(y_true, y_pred)

        grads = tape.gradient(loss, model.trainable_weights)

        opt.apply_gradients(zip(grads, model.trainable_weights))

        acc = metric.result()

        metric.reset_states()

        sleep(0.3)
        
        avg_loss = ((avg_loss * (stp - 1)) + loss) / stp
        avg_acc = ((avg_acc * (stp - 1)) + acc) / stp
    
        start += batch_size
        end += batch_size
        
        values = [('loss', loss), ('acc', acc)]

        pb.add(batch_size, values = values)
        
    return model, avg_loss, avg_acc
    
    
def val_step(model, X_path, y_path, img_shape, n_classes, batch_size, loss_fcn, metric):
    n_samples = len(X_path)
    n_steps = n_samples // batch_size
    
    pb = Progbar(n_samples)
    
    start = 0
    end = batch_size
    
    avg_loss = 0.0
    avg_acc = 0.0

    for stp in range(1, n_steps + 1):
        X_batch = load_data(X_path[start : end], img_shape[0:2])
        y_true = load_data(y_path[start : end], img_shape[0:2])
        y_true = one_hot(y_true, n_classes)
        
        y_pred = model(X_batch, training = False)
        loss = loss_fcn(y_true, y_pred)

        metric.update_state(y_true, y_pred)

        acc = metric.result()
        metric.reset_states()
        
        avg_loss = ((avg_loss * (stp - 1)) + loss) / stp
        avg_acc = ((avg_acc * (stp - 1)) + acc) / stp
        
        sleep(0.3)
        
        values = [('loss', loss), ('acc', acc)]
    
        pb.add(batch_size, values = values)

        start += batch_size
        end += batch_size
        
    return model, avg_loss, avg_acc