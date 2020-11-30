import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from models import MLP
from utils import get_dataset, plot_model_training, classes


# Preprocessing
images, labels = get_dataset()

# Creating model
mlp = MLP(images[0].shape, len(classes), 0.01)

# Training
epochs, hist = mlp.train(images, labels, epochs=1)
plot_model_training(epochs, hist)