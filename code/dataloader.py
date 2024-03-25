from param import dataset_path,sample_universe_size,train_size
from utils import prepare_data, mix_signals,load_audio,load_music
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
combination_paths = prepare_data(dataset_path)
sampled_combinations = combination_paths.sample(frac=sample_universe_size, random_state=42,ignore_index=True)
# sample_size = len(sampled_combinations) // 3

# sampled_combinations['sample_type'] = np.nan
# sampled_combinations.loc[:sample_size, 'sample_type'] = "speech"
# sampled_combinations.loc[sample_size:2*sample_size, 'sample_type'] = "music"
# sampled_combinations.loc[2*sample_size:, 'sample_type'] = "mixture"

# # Convert sample_type to integer type
# sampled_combinations['sample_type'] = sampled_combinations['sample_type'].astype(int)

class SignalDataset(Dataset):
    def __init__(self, sampled_df):
        self.sampled_df = sampled_df
        self.num_examples = len(sampled_df)
        self.data, self.label = self.load_data()

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def load_data(self):
        data_list = []
        label_list = []
        for row in range(self.sampled_df):
            for item in ["music","speech","mixture"]:
                if item == "mixture":
                    data = mix_signals(row['speech'], row['music'])
                    label = "mixture"
                else:
                    data = load_music(row[item])
                    label = item
                
                data_list.append(torch.tensor(data, dtype=torch.float32))
                label_list.append(label)

        return data_list,label_list