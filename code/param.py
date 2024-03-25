frame_size = 32000  # for example, 1 second of audio at 16 kHz
sampling_rate = 16000  # assuming both files have the same sampling rate
decibel_range = [-5,5]
dataset_path = 'data/musan'
types_allowed = ['music', 'speech']
sample_universe_size = 0.2 
train_size = 0.8