import PIL
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def get_audio(path):
    # Load the audio
    audio, sampling_rate = librosa.load(path)
    return audio, sampling_rate

def get_spectrogram(path):
    audio, sampling_rate = get_audio(path)
    
    # Get short time fourier transform
    spectrum = librosa.stft(audio, hop_length=512)
    spectrum = np.abs(spectrum)
    spectrum = librosa.amplitude_to_db(spectrum)

    return spectrum, sampling_rate

def display_spectrogram(spectrum, sampling_rate):
    librosa.display.specshow(spectrum, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

def matrix2image(mat):
    """
    Gets a 2D np.array with float values.
    Maps the values between 0 - 255.
    Returns a image
    """
    mat -= mat.min()
    mat /= mat.max()
    mat *= 255
    
    mat = np.array(mat, dtype='uint8')
    img = PIL.Image.fromarray(mat)
    return img


def save_spectrogram(spectrum, filename, method='image'):
    """
    Saves given spectrogram as an image or stores as a pickled file.
    """
    if 'image' == method:
        img = matrix2image(spectrum)
        img.save(filename + ".png")
    elif 'pickle' == method:
        np.save(filename, spectrum)

def load_spectrogram(filename):
    """
    Unpickles the stored file and loads as the spectrum as a np.array
    """
    spectrum = np.load(filename)
    return spectrum