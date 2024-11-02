from asteroid.models import BaseModel
import soundfile as sf
import numpy as np
import subprocess
import librosa
import os
import shutil
import time
import csv
import argparse
from pathlib import Path
    
def get_subfolders(folder_path):
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort()
    return subfolders

def mp4_to_wav(vid_path):
    output_audio = f'{vid_path[:-4]}.wav'

    print("mp4 to wav")
    
    # FFmpeg 명령어 실행
    command = ['ffmpeg', '-i', vid_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_audio]
    subprocess.call(command)
    return output_audio

def dptnet(model_dpt,audio):
    # audio : 'audio/{file_name}/{direction}/{file_name}_{i}.wav'
    model = BaseModel.from_pretrained("JorisCos/DPTNet_Libri1Mix_enhsingle_16k")    # asteroid의 pretrained model
    model_dpt.separate(audio, force_overwrite=True, resample = True)    # save as 'audio/{file_name}/{direction}/{file_name}_{i}_est1.wav'
    print("separate direct")
    dir_audio_path=f"{audio[:-4]}_est1.wav"
    return dir_audio_path

def separate_reverb(path):
    # Load mixed audio (contains speech and reverb)
    mixed_audio, sr_mixed = librosa.load(path[:-9]+".wav", sr=None)

    # Load direct audio
    direct_audio, sr_direct = librosa.load(path, sr=None)
    direct_audio = librosa.resample(direct_audio, orig_sr=sr_direct, target_sr=sr_mixed)
    
    # Compute STFT for mixed audio and direct audio
    n_fft = 1024
    hop_length = 512

    mixed_stft = librosa.stft(mixed_audio, n_fft=n_fft, hop_length=hop_length)
    direct_stft = librosa.stft(direct_audio, n_fft=n_fft, hop_length=hop_length)

    # Make sure the shapes are the same
    if mixed_stft.shape[1] != direct_stft.shape[1]:
        min_frames = min(mixed_stft.shape[1], direct_stft.shape[1])
        mixed_stft = mixed_stft[:, :min_frames]
        direct_stft = direct_stft[:, :min_frames]

    # Compute magnitude spectrograms
    mixed_mag = np.abs(mixed_stft)
    direct_mag = np.abs(direct_stft)

    # Perform Spectral Subtraction to separate reverb
    alpha = 1.0  # Adjustment parameter
    reverb_mag = mixed_mag - alpha * direct_mag

    # Apply phase of mixed audio to reverb magnitude
    reverb_stft = reverb_mag * np.exp(1j * np.angle(mixed_stft))

    # Inverse STFT to obtain separated reverb
    reverb_signal = librosa.istft(reverb_stft, hop_length=hop_length)

    # Save reverb audio
    sf.write(f'{path[:-9]}_r.wav', reverb_signal, sr_mixed, format='WAV')
    print("separate reverb")


def main():
    data_dir = "/home/gunwoo/kunwoolee/Past_project/Audio_Detection/DEEPFAKE_project/Data_split/splilt_1"
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     "data_dir",
    #     help = "Directory containing original data",
    #     type = str,
    #     default = "/home/gunwoo/kunwoolee/Past_project/Audio_Detection/DEEPFAKE_project/Data_split/splilt_1"
    # )
    # arg = parser.parse_args()
    model_dpt= BaseModel.from_pretrained("JorisCos/DPTNet_Libri1Mix_enhsingle_16k")
    data_paths = list(sorted(Path(data_dir).glob("**/*.wav")))
    
    
    for path in data_paths:
        dir_name = data_dir.split("/")[-1]
        file_name = str(path)
        target_dir = os.path.dirname(file_name).replace(f"{dir_name}", f"{dir_name}_dr")
        os.makedirs(target_dir, exist_ok =True)
        direct_audio_path = dptnet(model_dpt, file_name)
        separate_reverb(direct_audio_path)
        new_direct_path = f"{direct_audio_path[:-9]}_d.wav"
        reverb_path = f"{direct_audio_path[:-9]}_r.wav"
        os.rename(direct_audio_path , new_direct_path)
        shutil.move(new_direct_path , target_dir)
        shutil.move(reverb_path, target_dir)        



            
            
            
            
if __name__ == "__main__":
    main()

        


