# data Directory and Paths
data_dir = r"./data"
dataset_path = f"{data_dir}/musan"
sample_universe_size = 0.01

# Data Prep Parameters
types_allowed = ["music", "speech"]
n_mels = 128
frame_size = 32000  # for example, 1 second of audio at 16 kHz
sampling_rate = 16000  # assuming both files have the same sampling rate
decibel_range = [-5, 5]

# model Parameters
hyper_parameters = {
    "n_layers": 1,
    "sp_hidden_nodes": 20,
    "n_sp_hidden_lyrs": 1,
    "mu_hidden_nodes": 20,
    "n_mu_hidden_lyrs": 1,
    "smr_hidden_nodes": 20,
    "n_smr_hidden_lyrs": 1,
    "n_epochs": 100,
    "batch_size": 20,
    "train_ratio": 0.8,
}
model_dir = r"./model"
model_1 = f"{model_dir}/model_{sample_universe_size}.pth"
