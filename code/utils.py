import librosa
import torch
import torchaudio
import numpy as np
import random
import os
import pandas as pd
from itertools import product
# from dataloader import dataloader, SignalDataset
# from dataloader import SignalDataset
from param import (
    dataset_path,
    sample_universe_size,
    hyper_parameters as hp,
    data_file,
    sampling_rate,
    frame_size,
    decibel_range,
    types_allowed,
    targ_dict
)

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



def train(
    train_loader,
    model,
    epoch,
    out_dict,
    loss_sp_fn,
    loss_mu_fn,
    loss_smr_fn,
    optimizer,
    device,
):
    correct = 0
    for data in train_loader:
        feature, label = data
        feature = feature.to(device)
        y = [out_dict[x] for x in label]
        out_sp, out_mu, out_smr = model(feature)

        sp_list = [inner_list[0] for inner_list in y]
        mu_list = [inner_list[1] for inner_list in y]
        smr_list = [inner_list[2] for inner_list in y]
        y_sp = torch.Tensor(sp_list).unsqueeze(1).to(device)
        y_mu = torch.Tensor(mu_list).unsqueeze(1).to(device)
        y_smr = torch.Tensor(smr_list).unsqueeze(1).to(device)

        loss_sp = loss_sp_fn(out_sp, y_sp)
        loss_mu = loss_mu_fn(out_mu, y_mu)
        loss_smr = loss_smr_fn(out_smr, y_smr)

        total_loss = loss_sp + loss_mu + loss_smr
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pred_y = torch.cat((out_sp, out_mu, out_smr), dim=1)
        result = (pred_y > 0.5).float()
        target = torch.tensor(y).float().to(device)
        for i in range(result.size(0)):
            if torch.all(torch.eq(result[i], target[i])):
                correct += 1
    accuracy = correct / len(train_loader.dataset)
    print(
        f"Epoch: {epoch}, Loss_sp: {loss_sp}, Loss_mu: {loss_mu}, Loss_smr: {loss_smr}, Accuracy: {accuracy}"  # noqa
    )


def evaluate(model, test_loader, device):
    model.eval()
    su_predictions = []
    mu_predictions = []
    smr_predictions = []
    su_target = []
    mu_target = []
    smr_target = []
    for data in test_loader:
        feature, target = data
        su_pred,mu_pred,smr_pred = model(feature)
        su_predictions.append((su_pred  > 0.5).float())
        mu_predictions.append((mu_pred  > 0.5).float())
        smr_predictions.append((smr_pred  > 0.5).float())
        su_target.append(float(targ_dict[target[0]][0]))
        mu_target.append(float(targ_dict[target[0]][1]))
        smr_target.append(float(targ_dict[target[0]][2]))

    return su_predictions, mu_predictions, smr_predictions, su_target, mu_target, smr_target

def calculate_metrics(predictions, targets):
    # Convert predictions and targets to tensors if they're not already
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Calculate true positives, false positives, and false negatives
    true_positives = torch.sum(predictions * targets).float()
    false_positives = torch.sum(predictions * (1 - targets)).float()
    false_negatives = torch.sum((1 - predictions) * targets).float()

    # Calculate precision
    precision = true_positives / (true_positives + false_positives + 1e-10)

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = torch.sum(predictions == targets).float() / targets.numel()
    return precision.item(), recall.item(), f1_score.item(),accuracy.item()