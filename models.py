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

    def train(self, features, labels, batch_size=16, epochs=10, shuffle=True):
        history = self.model.fit(features, labels, batch_size, epochs, shuffle=shuffle)

        self.epochs = history.epoch
        self.hist = pd.DataFrame(history.history)

        return self.epochs, self.hist

    def test(self, features, labels):
        _, self.accuracy = self.model.evaluate(features, labels, verbose=0)

        return self.accuracy

    def predict(self, features):
        prediction = self.model(features)

        return prediction


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
