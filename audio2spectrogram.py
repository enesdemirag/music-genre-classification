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