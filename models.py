import tensorflow as tf
import pandas as pd


class MLP(object):
    def __init__(self, input_size, output_size, learning_rate):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Flatten(input_shape=input_size))
        self.model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=1024, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=256, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=output_size, activation="softmax"))

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["accuracy"]
        )


class CNN(object):
    def __init__(self, input_size, output_size, learning_rate):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation="relu", input_shape=input_size))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=32, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=output_size, activation="softmax"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
