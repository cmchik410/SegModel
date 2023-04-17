import numpy as np

def Reduce_LR(opt, scores, history, decay = 0.9, patience = 2, minimum = 1e-5, verbose = 0):
    temp = history[scores]
    
    if len(temp) <= patience:
        return None
    
    avg = np.mean(temp[-patience - 1 : -1])
    
    if avg - temp[-1] <= 0:
        current_lr = opt.learning_rate.numpy().tolist()
        new_lr = current_lr * decay
        if new_lr < minimum:
            new_lr = minimum
        
        opt.learning_rate.assign(new_lr)
        
        if verbose:
            print("\n Learning Rate updated. Current learning rate is %f" %(new_lr))
        

def get_best_model(scores, history, patience = 2, verbose = 0):
    temp = history[scores]
    
    if len(temp) <= patience:
        return False
    
    avg = np.mean(temp[-patience - 1 : -1])
    
    if avg - temp[-1] <= 0:
        if verbose:
            print("\n Retrieve best model")
        return True
    
    return False
    