import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_training(epochs, hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    x = hist['accuracy']
    plt.plot(epochs[1:], x[1:], label='accuracy')
    plt.legend()
    plt.show()

def save_model(model, path):
    model.save(path)

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model