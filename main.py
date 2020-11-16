from audio2spectrogram import *
from train import * 
from test import *
from models import *
from preprocessing import *

# Preprocessing
spec, fs = get_spectrogram('dataset/audios/blues.wav')
img = matrix2image(spec)

# Creating models
mlp = MLP(spec.shape, len(classes), 0.01)

# Prediction
y = predict(mlp, img)
print(y)