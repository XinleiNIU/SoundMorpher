import os
os.environ["CURL_CA_BUNDLE"]=""
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Import AudioLDM2
import scipy
import IPython.display as ipd
import torchaudio
import yaml
import audioldm_train.utilities.audio as Audio
from librosa.filters import mel as librosa_mel_fn
from torch.cuda.amp import autocast, GradScaler
import librosa

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from diffuser_helpers_cond_uncond_lora import extract_lora_diffusers, predict_noise0_diffuser
# from lpmc.music_captioning import captioning
from preprocessor import Preprocessor
from diffusers.utils.torch_utils import randn_tensor
from scipy.io.wavfile import write
from IPython.display import display, clear_output
import cdpam
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import ast
import os
from audioldm_eval import EvaluationHelper
import argparse
import pdb
device = "cuda:0"
import pandas as pd
import torch
import ast



def process_audio_pair(input_dir, a_prev, a_next):
    wav_ref = cdpam.load_audio(os.path.join(input_dir, a_prev))
    wav_out = cdpam.load_audio(os.path.join(input_dir, a_next))
    return wav_ref, wav_out

def main(input_dir, target_audio_path):

    # Obtain list of subfolder
    subfolder = os.listdir(input_dir)

    loss_fn = cdpam.CDPAM()
    evaluator = EvaluationHelper(16000, device)


    # Define the table headers
    headers = ["Mean CDPAM", "STD CDPAM", "Total CDPAM", "Error CDPAM", "MFCCs error","FAD","FD"]
    # Create an empty DataFrame
    df = pd.DataFrame()

    with torch.no_grad():
        # iteratively eval
        for sf in tqdm(subfolder):
            # Define eval metric list
            eval_metric = []
            # Read alpha list
            with open(os.path.join(input_dir,sf,'alpha_list.txt'),'r') as file:
                content = file.read()
                alpha_list = ast.literal_eval(content)
                input_sample_1 = os.path.join(target_audio_path, sf.split("_")[0])
                input_sample_2 = os.path.join(target_audio_path, sf.split("_")[1])
            audio_dir = os.path.join(input_dir,sf,"audios")

            # Calculate CDPAM
            CDPAM_list = []
            for (i,j) in zip(alpha_list,alpha_list[1:]):
                a_prev = f'{i}.wav'
                a_next = f'{j}.wav'
                try:
                    wav_ref, wav_out = process_audio_pair(audio_dir, a_prev, a_next)
                    
                    # Perform the forward pass and calculate the distance
                    dist = loss_fn.forward(wav_ref, wav_out)
                    CDPAM_list.append(dist.item())
                    
                    # Clear cache after each iteration
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"Skipping pair ({a_prev}, {a_next}) due to out of memory error")
                        torch.cuda.empty_cache()
                    else:
                        raise e
            eval_metric.append(np.mean(CDPAM_list))
            eval_metric.append(np.std(CDPAM_list))
            eval_metric.append(np.sum(CDPAM_list))

            torch.cuda.empty_cache()   
            dist_total = 0
            # sample 1 and reconstructed sample 1
            wav_ref = cdpam.load_audio(input_sample_1)
            wav_out = cdpam.load_audio(f'{audio_dir}/{1}.wav')
            dist_total += loss_fn.forward(wav_ref,wav_out)
            wav_ref = cdpam.load_audio(input_sample_2)
            wav_out = cdpam.load_audio(f'{audio_dir}/{0}.wav')
            dist_total += loss_fn.forward(wav_ref,wav_out)
            eval_metric.append(dist_total.cpu().detach().numpy()[0]/2)
            torch.cuda.empty_cache()

            # MFCCs
            # Load audio files
            y1, sr1 = librosa.load(input_sample_1)
            y2, sr2 = librosa.load(input_sample_2)

            # Compute MFCCs
            mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
            mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)


            # Find the middle content point
            i = alpha_list[int((len(alpha_list)+1)/2)-1]

            morph_sample = f'{i}.wav'
            ymorphed, srmorphed = librosa.load(f'{audio_dir}/{morph_sample}')
            mfcc_morphed = librosa.feature.mfcc(y=ymorphed, sr=srmorphed)

            # Compute distance metrics between morphed and inputs
            dist_mfcc_1 = np.mean(np.linalg.norm(mfcc_morphed - mfcc1, axis=0))
            dist_mfcc_2 = np.mean(np.linalg.norm(mfcc_morphed - mfcc2, axis=0))

            eval_metric.append(np.abs(dist_mfcc_1/(dist_mfcc_1+dist_mfcc_2)-0.5))
            torch.cuda.empty_cache()
            metrics = evaluator.main(audio_dir,target_audio_path,limit_num=None)
            # Store the list into pd table
            eval_metric.append(metrics['frechet_audio_distance'])
            eval_metric.append(metrics['frechet_distance'])

            df = pd.concat([df, pd.DataFrame([eval_metric])], ignore_index=True)
    means = df.mean()
    result = pd.DataFrame([means.values], columns=headers)
    print(result)

    result.to_excel(os.path.join(input_dir,'eval.xlsx'), index=False)
    print("Completed eval and wrote into path:",os.path.join(input_dir,'eval.xlsx'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--morph_dir", type=str, required=True, help ="path to morphed results")
    parser.add_argument("--source_dir",  type=str, required=True, help ="path to sourced results")
    args = parser.parse_args()


    input_dir = args.morph_dir
    # Sourced audio path, should contains sourced audios only.
    target_audio_path = args.source_dir

    # evaluate
    main(input_dir,target_audio_path)