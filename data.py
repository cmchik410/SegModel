import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def main():
    train_dataset = tfds.load("scene_parse150", split=tfds.Split.TRAIN, as_supervised=True)
    
    dataset = train_dataset.map(lambda img, label: (tf.image.resize(img, [64, 64]), label))

    X_train = []
    y_train = []

    for images, labels in dataset:
        X_train.append(images)
        y_train.append(labels)
        
    print(X_train.shape)
    print(y_train.shape)
    
main()