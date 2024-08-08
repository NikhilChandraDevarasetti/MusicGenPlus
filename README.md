# MusicGenPlus
MusicGenPlus is an improved version of MusicGen small model. The Final CLAP score on MusicCaps Data surpassed small, medium and large models of MusicGen.


## Installation
```bash
pip install -r new_requirements.txt
```
It is also recommended having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```
## Dataset
Musicbench Training Data: https://drive.google.com/drive/folders/1-fFYHhvlcshWSWg3I271mXq26VcQR7Qm?usp=sharing

Musiccaps Test Data: https://drive.google.com/drive/folders/1Eu7uGptiU1xm3_9iinfBJfCd4LEjLS50?usp=sharing

Generated Audios: 
## Models

MusicGenPlus required model weights: https://drive.google.com/drive/folders/1HZdHtFryZbPVHVGcgVnulKQUHFlVoZYA?usp=sharing

## Training

Create a folder, in it, place your audio and caption files. They must be .wav and .txt format respectively. You can omit .txt files for training with empty text by setting the --no_label option to 1.

![68747470733a2f2f692e696d6775722e636f6d2f416c446c7142492e706e67](https://github.com/user-attachments/assets/25af6592-83c6-440d-a6cb-d758229bea84)



You can use .wav files longer than 30 seconds, in that case the model will be trained on random crops of the original .wav file.

In this example, segment_000.txt contains the caption "jazz music, jobim" for wav file segment_000.wav.

## Running the trainer
Run 
```bash
python3 run.py --dataset <PATH_TO_YOUR_DATASET>
```
Make sure to use the full path to the dataset, not a relative path.

Once training finishes, the model (and checkpoints) will be available under the models folder in the same path you ran the trainer on.

To load them, simply run the following on your generation script:

```bash
model.lm.load_state_dict(torch.load('models/lm_final.pt'))
```
Where model is the MusicGen Object and models/lm_final.pt is the path to your model (or checkpoint).




