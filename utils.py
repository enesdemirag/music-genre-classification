import PIL
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

classes = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}


def load_audio(path): # Loads audio from file
    audio, sampling_rate = librosa.load(path)
    return audio, sampling_rate


def load_spectrogram(path): # Loads matrix from file
    """
    Unpickles the stored file and loads as the spectrum as a np.array
    """
    spectrum = np.load(path)
    return spectrum


def create_spectrogram(audio, sampling_rate, method="mel"): 
    # Get short time fourier transform
    if "linear" == method:
        spectrum = librosa.stft(audio, hop_length=512)
        spectrum = np.abs(spectrum)
        spectrum = librosa.amplitude_to_db(spectrum)
        return spectrum
    elif "mel" == method:
        spectrum = librosa.feature.melspectrogram(audio, sampling_rate, n_mels=128, n_fft=2048, hop_length=1024)
        spectrum = librosa.power_to_db(spectrum, ref=np.max)
        return spectrum


def display_spectrogram(spectrum, sampling_rate, scale="mel"): 
    """
    Frequency types     : 
    ‘linear’ , ‘fft’,     ‘hz’: frequency range is determined by the FFT window and sampling rate.
              ‘log’     : the spectrum is displayed on a log scale.
              ‘mel’     : frequencies are determined by the mel scale.
              ‘cqt_hz’  : frequencies are determined by the CQT scale.
              ‘cqt_note’: pitches are determined by the CQT scale.
    """
    librosa.display.specshow(spectrum, sr=sampling_rate, x_axis="time", y_axis=scale)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.show()


def matrix2image(mat, scale="linear"): 
    """
    Gets a 2D np.array with float values.
    Maps the values between 0 - 255.
    Returns a image
    """
    mat -= mat.min()
    mat /= mat.max()
    mat *= 255

    mat = np.array(mat, dtype="uint8")

    if "linear" == scale:
        return PIL.Image.fromarray(mat)
    elif "log"  == scale:
        pass


def save_spectrogram(spectrum, filename, method="image"): 
    """
    Saves given spectrogram as an image or stores as a pickled file.
    """
    if "image" == method:
        img = matrix2image(spectrum)
        img.save(filename + ".png")
    elif "pickle" == method:
        np.save(filename, spectrum)


def plot_model_training(epochs, hist): 
    x = hist["accuracy"]
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.plot(epochs[1:], x[1:], label="accuracy")
    plt.legend()
    plt.show()


def load_model(path): 
    return tf.keras.models.load_model(path)
