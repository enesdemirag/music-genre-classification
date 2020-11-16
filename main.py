from train import * 
from test import *
from models import *
from preprocessing import *

# Preprocessing
spec, fs = get_spectrogram('dataset/audios/blues.wav')
img = matrix2image(spec)

# Creating models
mlp = MLP(spec.shape, len(classes), 0.01)
# cnn = CNN(spec.shape, len(classes), 0.01)

# Training
# epochs, hist = train_model(mlp, train_data, labels, epochs=10)
# plot_training(epochs, hist)

# Testing
# accuracy = test_model(mlp, test_data, labels)
# print(result)

# Prediction
y = predict(mlp, img)
print(y)