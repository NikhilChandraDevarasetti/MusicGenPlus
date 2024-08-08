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


def generate_and_save_audio(description, wav_file_name, output_dir):
    # Generate audio
    res = model.generate([description], progress=True)

    # Convert the generated tensor to a NumPy array
    generated_audio = res[0].cpu().numpy()

    # Save audio
    audio_output_path = os.path.join(output_dir, wav_file_name)
    sf.write(audio_output_path, generated_audio.T, 48000)  # Save with the same sample rate

    return res


# Directories
music_data_dir = '/home/nikhil/Documents/yeshiva_projects/audiocraft/musiccaps_training_data'
output_dir = 'Bulk_Generated_audios'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize audio tensor
audio_tensor = torch.empty(0)

# Process all files
files_processed = 0
for file_name in os.listdir(music_data_dir):
    if file_name.endswith('.wav'):
        wav_file_path = os.path.join(music_data_dir, file_name)
        description_file_path = wav_file_path.replace('.wav', '.txt')
        audio_output_path = os.path.join(output_dir, file_name)

        # Check if the generated file already exists
        if os.path.isfile(audio_output_path):
            print(f"File {audio_output_path} already exists. Skipping.")
            continue

        if os.path.isfile(description_file_path):
            # Read description with UTF-8 encoding
            with open(description_file_path, 'r', encoding='utf-8') as file:
                description = file.read().strip()

            res = generate_and_save_audio(description, file_name, output_dir)
            files_processed += 1

            # Append the generated audio tensor
            if audio_tensor.size(0) == 0:
                audio_tensor = res
            else:
                audio_tensor = torch.cat([audio_tensor, res], dim=0)
                
            torch.save(audio_tensor, 'audio_tensor.pth')
    print("Files processed: ", files_processed)

# Save the concatenated tensor
torch.save(audio_tensor, 'audio_tensor.pth')

print("Total Files Processed:", files_processed)
print("Generated audio and saved descriptions for new files in:", output_dir)
