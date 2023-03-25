import yaml

def read_config(fpath):
    with open(fpath, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        
    kwargs = {}
    
    kwargs["train_path"] = config["DATA"]["train"]
    kwargs["train_label_path"] = config["DATA"]["train_labels"]
    kwargs["val_path"] = config["DATA"]["val"]
    kwargs["val_label_path"] = config["DATA"]["val_labels"]
    kwargs["color_dict"] = config["DATA"]["color_dict"]
    kwargs["dimensions"] = tuple(config["DATA"]["dimensions"])
    kwargs["channels"] = config["DATA"]["channels"]
    kwargs["n_classes"] = config["DATA"]["n_classes"]
    
    kwargs["ppm_output_channel"] = config["TRAIN"]["ppm_output_channel"]
    kwargs["pool_size"] = tuple(config["TRAIN"]["pool_size"])
    kwargs["strides"] = config["TRAIN"]["strides"]
    kwargs["batch_size"] = config["TRAIN"]["batch_size"]
    kwargs["epochs"] = config["TRAIN"]["epochs"]
    kwargs["learning_rate"]  = config["TRAIN"]["learning_rate"]
    
    kwargs["model_path"]  = config["END"]["model_path"]
    
    return kwargs
    