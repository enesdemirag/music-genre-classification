from audio2spectrogram import *

spec, fs = get_spectrogram('dataset/audios/blues.wav')
save_spectrogram(spec, 'dataset/spectrograms/blues', 'image')

