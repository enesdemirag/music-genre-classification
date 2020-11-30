"""
This script gets all audio files inside in_path,
and transforms them into spectrograms,
then saves them as images inside out_path.
"""

import os
import csv
from utils import classes, load_audio, audio2spectrogram, extract_features

in_path = "dataset/audios/"
out_path = "dataset/spectrograms/"

# Spectrogram Data
def init_image_dataset():
    for genre in classes.values():
        # Create output directory
        if not os.path.exists(out_path + genre):
            os.mkdir(out_path + genre)

        # Get all audio files
        files = os.listdir(in_path + genre)

        for f in files:
            # Define paths
            audio_path = in_path + genre + "/" + f
            spec_path = out_path + genre + "/" + f

            # Load audio and create spectrogram
            audio, fs = load_audio(audio_path)
            audio2spectrogram(audio, fs, spec_path)
            print("Saved:", spec_path)


# Analytical Data
def init_analytics_dataset():
    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        header = "chroma_freqs spectral_centroid spectral_bandwidth spectral_rolloff zero_crossing_rate"
        for i in range(1, 21):
            header += " mfcc" + str(i)
        header += " genre"
        writer.writerow(header.split())
        
        # Writing Data
        for genre in classes.values():
            # Get all audio files
            files = os.listdir(in_path + genre)

            for f in files:
                audio, fs = load_audio(in_path + genre + "/" + f)
                features = extract_features(audio, fs)
                features.append(genre)
                writer.writerow(features)
                print("Features extracted:", f)

# init_image_dataset()
init_analytics_dataset()