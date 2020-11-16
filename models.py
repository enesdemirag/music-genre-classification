import tensorflow as tf

def MLP(input_size, output_size, learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=input_size))
    model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=output_size, activation='softmax'))     
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    return model