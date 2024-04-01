from param import dataset_path,sample_universe_size,sampling_rate
from utils import mix_signals,load_audio,load_music
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from librosa.feature import melspectrogram
from librosa import power_to_db
from multiprocessing import Pool
from tqdm import tqdm

# import time
# start_time = time.time()


# Define the SignalDataset class to store the data in required format
class SignalDataset(Dataset):
    def __init__(self, sampled_df, data_type):
        
        self.sampled_df = sampled_df
        self.num_examples = len(sampled_df)
        if data_type not in ["music","speech","mixture"]:
            raise ValueError(f"Data type {data_type} not allowed. Please choose from ['music','speech','mixture']")
        self.type = data_type
        self.data, self.label = self.load_data()

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def process_row(self, args):
        row, type_ = args
        if type_ == "mixture":
            return self.get_spectogram_mix(row)
        else:
            return self.get_spectogram_signal(row)
    
    def load_data(self):
        lable_list = [self.type]*len(self.sampled_df)
        data_list = []

        with Pool() as p:
            data_list = list(tqdm(p.imap(self.process_row, [(row, self.type) for _, row in self.sampled_df.iterrows()]), total=len(self.sampled_df)))

        return data_list, lable_list
    
    def get_spectogram_mix(self,row):
        data_list = []
        label_list = [self.type]*len(self.sampled_df)
        data = mix_signals(row['speech'], row['music'])   
        mel_spectrogram = melspectrogram(y=data,sr=sampling_rate)
        spect_decib = power_to_db(mel_spectrogram, ref=np.max)
        spec_tensor = torch.from_numpy(spect_decib)
        return spec_tensor
    
    def get_spectogram_signal(self,row):
        data = load_music(row[self.type])
        mel_spectrogram = melspectrogram(y=data,sr=sampling_rate)
        spect_decib = power_to_db(mel_spectrogram, ref=np.max)
        spec_tensor = torch.from_numpy(spect_decib)
        return spec_tensor
        


# Function to load the data for training and testing
def dataloader(datasets,train_ratio = 0.8,train_batch_size =20,test_batch_size =20):
    train_size = int(train_ratio * len(datasets))
    test_size = len(datasets) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])
    train_loader =DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    return train_loader,test_loader

#Call this code in train and evaluate functions (to be deleted later)
# combination_paths = prepare_data(dataset_path)
# sampled_df = combination_paths.sample(frac=sample_universe_size, random_state=42,ignore_index=True)

# datasets_music = SignalDataset(sampled_df, data_type="music")
# train_loader_music,test_loader_music = dataloader(datasets=datasets_music,train_ratio = 0.8,train_batch_size =20,test_batch_size =20)

# datasets_speech = SignalDataset(sampled_df, data_type="speech")
# train_loader_speech,test_loader_speech = dataloader(datasets=datasets_speech,train_ratio = 0.8,train_batch_size =20,test_batch_size =20)

# datasets_mixture = SignalDataset(sampled_df, data_type="mixture")
# train_loader_mixture,test_loader_mixture = dataloader(datasets=datasets_mixture,train_ratio = 0.8,train_batch_size =20,test_batch_size =20)

# end_time = time.time()

# total_time = end_time - start_time
# print(f"Total runtime of the script is {total_time} seconds")