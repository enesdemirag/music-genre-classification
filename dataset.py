"""
This script gets all audio files inside in_path,
and transforms them into spectrograms,
then saves them as images inside out_path.
"""

from preprocessing import classes
from audio2spectrogram import *
import os

in_path = "dataset/audios/"
out_path = "dataset/spectrograms/"

for genre in classes.values():
    # Create output directory
    if not os.path.exists(out_path + genre):
        os.mkdir(out_path + genre)

    # Get all audio files
    files = os.listdir(in_path + genre)

    for f in files:
        # Define paths
        audio_path = in_path + genre + '/' + f
        spec_path = out_path + genre + '/' + f

        # Load audio and create spectrogram
        audio, fs = load_audio(audio_path)
        spec = create_spectrogram(audio)

        # Save spectrogram image
        save_spectrogram(spec, spec_path)

        print("Saved:", spec_path)
