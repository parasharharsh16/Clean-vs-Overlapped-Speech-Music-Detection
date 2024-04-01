import librosa
import torch
import torchaudio
import numpy as np
import random
import os
import pandas as pd
from itertools import product
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.decomposition import PCA
from param import (
    dataset_path,
    sample_universe_size,
    hyper_parameters as hp,
    data_file,
    sampling_rate,
    frame_size,
    decibel_range,
    types_allowed,
    targ_dict,classical_model_path_rf,
    classical_model_path_svm,
    combination_file_path
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

def prepare_data(dataset_path,combination_file_path):
    if os.path.exists(combination_file_path):
        df_combined = pd.read_csv(combination_file_path)
        return df_combined
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
    df_combined.to_csv(combination_file_path, index=False)

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

def prediction_thresold(data):
    if round(np.mean(data),2) > 0:
        prediction = 'music'
    elif round(np.mean(data),2) > -2.5:
        prediction = 'mixture'
    else:
        prediction = 'speech'
    return prediction
        

def classical_classification(csv_filepath):
    sampled_df = prepare_data(csv_filepath,combination_file_path).sample(frac=sample_universe_size, random_state=42,ignore_index=True)
    for i in range(len(sampled_df)):
        speech_wave = load_music(sampled_df.iloc[i]["speech"])
        music_wave = load_music(sampled_df.iloc[i]["music"])
        mixed_wave = mix_signals(sampled_df.iloc[i]["speech"],sampled_df.iloc[0]["music"])
        #hpss_classification(speech_wave,music_wave,mixed_wave,sampling_rate)
        harmonic_speech,_ = librosa.effects.hpss(speech_wave)
        harmonic_music, _ = librosa.effects.hpss(music_wave)
        harmonic_mixture, _ = librosa.effects.hpss(mixed_wave)
        label_dict = {"speech": 0, "music": 1, "mixture": 2}
        su_groundTruth = []
        mu_groundTruth = []
        smr_groundTruth = []
        su_predictions = []
        mu_predictions = []
        smr_predictions = []
        for ground_truth in ['music','mixture','speech']:
            if ground_truth == 'music':
                data = harmonic_music
                su_groundTruth.append(label_dict[ground_truth])
                su_predictions.append(label_dict[prediction_thresold(data)])
            elif ground_truth == 'mixture':
                data = harmonic_mixture
                mu_groundTruth.append(label_dict[ground_truth])
                mu_predictions.append(label_dict[prediction_thresold(data)])
            elif ground_truth == 'speech':
                data = harmonic_speech
                smr_groundTruth.append(label_dict[ground_truth])
                smr_predictions.append(label_dict[prediction_thresold(data)])

    return su_predictions, mu_predictions, smr_predictions, su_groundTruth, mu_groundTruth, smr_groundTruth

def train_classical_model(train_loader,model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC()
        classical_model_path = classical_model_path_svm
    elif model_type == 'rf':
        clf =  RandomForestClassifier()
        classical_model_path = classical_model_path_rf
    else:
        raise ValueError("Invalid model type. Please choose either 'svm' or 'rf'")
    # Loop over the data in the train_loader
    label_dict = {"speech": 0, "music": 1, "mixture": 2}
    feature = []
    target = []
    for data, labels in train_loader:
        #flatten the data
        data_flat = data.view(data.size(0), -1)
        #usie lable list to convert the labels to integers
        labels_int = [label_dict[label] for label in labels]
        feature.extend(data_flat)
        target.extend(labels_int)

    feature = np.array(feature)
    target = np.array(target)
    

    #Applying PCA for dimentionality reduction
    pca = PCA(n_components=2) 
    feature = pca.fit_transform(feature)
    # Train the SVM classifier
    clf.fit(feature, target)
    # Save the trained SVM classifier
    dump(clf, classical_model_path) 

    return clf

def evaluate_ml_clf(test_loader, clf):
    label_dict = {"speech": 0, "music": 1, "mixture": 2}

    feature = []
    target = []

    for data, labels in test_loader:
        #flatten the data
        data_flat = data.view(data.size(0), -1)
        #usie lable list to convert the labels to integers
        labels_int = [label_dict[label] for label in labels]
        feature.extend(data_flat)
        target.extend(labels_int)
    
    
    
    feature = np.array(feature)
    target = np.array(target)
    pca = PCA(n_components=2) 
    feature = pca.fit_transform(feature)

    su_groundTruth = []
    mu_groundTruth = []
    smr_groundTruth = []
    su_predictions = []
    mu_predictions = []
    smr_predictions = []
    for i in range(len(target)):
        if target[i] == 0:
            su_groundTruth.append(1)
            su_predictions.append(clf.predict(feature[i].reshape(1, -1))[0])

        elif target[i] == 1:
            mu_groundTruth.append(1)
            mu_predictions.append(clf.predict(feature[i].reshape(1, -1))[0])
        else:
            smr_groundTruth.append(1)
            smr_predictions.append(clf.predict(feature[i].reshape(1, -1))[0])
    return su_predictions, mu_predictions, smr_predictions, su_groundTruth, mu_groundTruth, smr_groundTruth


def plot_ROC_AUC_Curve(predictions, targets, class_name, output_folder):
    predictions = np.squeeze(predictions)
    targets = np.squeeze(targets)
    fpr, tpr, thresholds = roc_curve(predictions, targets)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.title(f'ROC Curve for {class_name}')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.fill_between(fpr, tpr, color='darkorange', alpha=0.2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'{output_folder}/roc_curve_{class_name}_{sample_universe_size}.png', )
    print(f"Saved ROC AUC curve for class: {class_name} as roc_curve_{class_name}_{sample_universe_size}.png")
    plt.close()
    

