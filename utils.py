import os
from PIL import Image
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
    audio, sampling_rate = librosa.load(path, mono=True)
    return audio, sampling_rate


def load_spectrogram(path, method="nparray"): # Loads matrix from file
    """
    Unpickles the stored file and loads as the spectrum as a np.array
    """
    if "pickle" == method:
        return np.load(path)
    elif "image" == method:
        return Image.open(path).convert('RGB')
    elif "nparray" == method:
        return np.array(Image.open(path).convert('RGB'))


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
        return Image.fromarray(mat)
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


def audio2spectrogram(audio, sampling_rate, path):
    plt.figure(figsize=(2.56, 2.56))
    plt.specgram(x=audio, NFFT=2048, Fs=sampling_rate, Fc=0, noverlap=128,
                 cmap=plt.get_cmap("inferno"), sides="default",
                 mode="psd", scale="dB")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path + ".png")
    plt.clf()
    plt.close()


def get_dataset():
    path   = "dataset/spectrograms/"
    images = np.zeros((1000, 256, 256, 3))
    labels = np.zeros((1000, 1))
    i      = 0
    
    for v, g in classes.items():
        files = os.listdir(path + g)
        for f in files:
            img = load_spectrogram(path + g + '/' + f)
            images[i] = img
            labels[i] = v
            i += 1
    return images, labels


def extract_features(y, sr):
    """ Features:
    - Chrome Frequencies
    - Root Mean Square Error
    - Spectral Centroid
    - Spectral Bandwith
    - Spectral Rolloff
    - Zero Crossing Rate
    - Mel-Frequecy Cepstral Coefficients (1 to 20)
    """
    chroma    = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc      = librosa.feature.mfcc(y=y, sr=sr)
    features  = [chroma, spec_cent, spec_bw, rolloff, zcr]    
    
    for e in mfcc:
        features.append(np.mean(e))
    
    return features

def load_analytic_data(path):
    """
    Loads analitic data from csv file and returns as a Pandas dataframe object.
    """
    return pd.read_csv(path) 