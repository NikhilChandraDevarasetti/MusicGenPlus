import os
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.metrics import CLAPTextConsistencyMetric, FrechetAudioDistanceMetric, KLDivergenceMetric, PasstKLDivergenceMetric
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import soundfile as sf
    
from audiocraft.modules.conditioners import (
    ClassifierFreeGuidanceDropout
)

import wandb

torch.cuda.empty_cache()
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.lm.load_state_dict(torch.load('models/lm_final_music_bench_5000.pt'))
model.set_generation_params(duration=30)

# def read_text_files_in_folder(folder_path):
#     descriptions = []
#     filenames = []
    
#     # Iterate over all files in the specified folder
#     for filename in os.listdir(folder_path):
#         # Check if the file is a text file and not a directory
#         if filename.endswith(".txt") and not os.path.isdir(os.path.join(folder_path, filename)):
#             filenames.append(filename.split('.txt')[0])  # Add only .txt filenames to the list
#             file_path = os.path.join(folder_path, filename)
            
#             # Open and read the text file
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 text = file.read()
#                 descriptions.append(text)

#     return descriptions, filenames

# # Specify the folder path
# folder_path = '/home/nikhil/Documents/yeshiva_projects/audiocraft/musiccaps_training_data'


# # Prompts
# # 'This is a nice guitar music', 
# descriptions, filenames = read_text_files_in_folder(folder_path)

import ast

# Open the file in read mode
with open('descriptions.txt', 'r', encoding='utf-8') as file:
    # Read the entire file content
    text_info = file.read()


with open('filenames.txt', 'r') as file:
    # Read the entire file content
    names = file.read()


# Use ast.literal_eval to safely evaluate the content
descriptions = ast.literal_eval(text_info)
filenames = ast.literal_eval(names)

file_description_dict = dict(zip(filenames, descriptions))

#print(descriptions)
#print(filenames)
#['Drums', 'Guitar', 'Jazz', 'voilin']

# Specify the output folder
output_folder = '/home/nikhil/Documents/yeshiva_projects/audiocraft/musiccaps_generated_audios'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

sample_rates = []
               

wav = model.generate(descriptions[300:305]).cpu()
# Example of using permute to correct dimensions
# wav = wav.permute(1, 0, 2).reshape(wav.size(1), -1)  # Adjust according to your needs

print(wav.size())
print(wav)
sample_rate = 48000

# Try to load existing tensors
try:
    existing_tensors = torch.load('tensor_data.pth')
    if not isinstance(existing_tensors, torch.Tensor):
        raise ValueError("Loaded data is not a tensor.")
except FileNotFoundError:
    # If file not found, initialize as an empty tensor
    existing_tensors = torch.empty((0,) + wav.shape[1:])

# Concatenate existing tensors with new tensor
extended_tensor = torch.cat([existing_tensors, wav], dim=0)

# Save the concatenated tensor
torch.save(extended_tensor, 'tensor_data.pth')


for idx, audio in enumerate(wav):
    filename = filenames[idx+300]
    audio_name = f"{filename}.wav"
    audio_path = os.path.join(output_folder, audio_name)
    torchaudio.save(audio_path, audio, sample_rate)
    sample_rates.append(sample_rate)
            
    sample_rates_tensor= torch.tensor(sample_rates)
#generated_wavs_tensor = torch.tensor(generated_wavs)

clap_metric = CLAPTextConsistencyMetric(model_path='music_speech_epoch_15_esc_89.25.pt', model_arch = 'HTSAT-base', enable_fusion=False).cpu()
# clap_metric.update(generated_wavs_tensor, descriptions, generated_wavs_tensor.size(), sample_rates_tensor)
clap_metric.update(wav, descriptions[10+15], wav.size(), sample_rates_tensor)
score = clap_metric.compute()
print(f"CLAP Score: {score}")

