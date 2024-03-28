import librosa
import torchaudio
import numpy as np
import random
from param import sampling_rate, frame_size,decibel_range,types_allowed,dataset_path
import os
import pandas as pd
from itertools import product


def change_volume(audio_signal, sr, change_db):
    # Calculate the amplitude ratio from the decibel change
    amplitude_ratio = 10 ** (change_db / 20)
    # Apply the amplitude ratio to the audio signal
    return audio_signal * amplitude_ratio

def normalize_audio(audio_signal):
    # Find the maximum absolute value in the signal
    max_val = np.max(np.abs(audio_signal))
    # Scale the signal so that the maximum absolute value is 1.0
    normalized_signal = audio_signal / max_val
    return normalized_signal
def mix_signals(speech_audio_path, music_audio_path):
    # Load the music and speech files
    music, sr_music= load_audio(music_audio_path)
    speech, sr_speech = load_audio(speech_audio_path)

    db_change_music = random.uniform(*decibel_range)
    db_change_speech = random.uniform(*decibel_range)

    # Change the volume of music and speech
    music = change_volume(music, sr_music, db_change_music)
    speech = change_volume(speech, sr_speech, db_change_speech)

    music = normalize_audio(music)
    speech = normalize_audio(speech)

    # Select a random frame from the music file
    start_frame = random.randint(0, len(music) - frame_size)
    music_frame = music[start_frame:start_frame + frame_size]

    # If the speech file is shorter than the music frame, pad it with zeros
    if len(speech) < frame_size:
        speech = np.pad(speech, (0, max(0, frame_size - len(speech))), 'constant')

    # Mix the music frame with the speech signal
    mixed_signal = (music_frame + speech[:frame_size]) / 2
    return mixed_signal

def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    # Convert the mixed signal to mono if it's stereo using torchaudio
    if audio.ndim == 2:
        audio = torchaudio.transforms.DownmixMono(channels_first=True)(audio)
        print("Converted stereo audio to mono")
    return audio, sr

def prepare_data(dataset_path):
    data_list = []
    for type_folder in os.listdir(dataset_path):
        if type_folder in types_allowed:
            type_folder_path = os.path.join(dataset_path, type_folder)
            for sub_folder in os.listdir(type_folder_path):
                sub_folder_path = os.path.join(type_folder_path, sub_folder)
                if(os.path.isdir(sub_folder_path)):
                    dict_files = [{"path": os.path.join(sub_folder_path, f), "type": type_folder} for f in os.listdir(sub_folder_path) if f.endswith('.wav')]
                    data_list.extend(dict_files)
    df_data = pd.DataFrame(data_list)
    df_speech = df_data[df_data['type'] == 'speech']
    df_music = df_data[df_data['type'] == 'music']
    # Generate all combinations of music and speech paths
    combinations = product(df_music['path'], df_speech['path'])
    df_combined = pd.DataFrame(combinations, columns=['music', 'speech'])
    df_unique_combinations =  df_combined.drop_duplicates(subset=['music', 'speech'], keep=False)
    # df_combined.to_csv('data/speech_music_combinations.csv', index=False)

    return df_combined

def load_music(music_audio_path):
    # Load the music and speech files
    music, sr_music= load_audio(music_audio_path)
    db_change_music = random.uniform(*decibel_range)
    # Change the volume of music and speech
    music = change_volume(music, sr_music, db_change_music)
    music = normalize_audio(music)

    # Select a random frame from the music file
    start_frame = random.randint(0, len(music) - frame_size)
    music_frame = music[start_frame:start_frame + frame_size]

    return music_frame
