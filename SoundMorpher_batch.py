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

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from diffuser_helpers_cond_uncond_lora import extract_lora_diffusers, predict_noise0_diffuser
from preprocessor import Preprocessor
from diffusers.utils.torch_utils import randn_tensor
from scipy.io.wavfile import write
from IPython.display import display, clear_output

import librosa
import numpy as np
from scipy.spatial.distance import euclidean
# import ast
import os
import argparse
from torch import lerp
import warnings

# Suppress all FutureWarnings
warnings.simplefilter("ignore", FutureWarning)



import random
import os
def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
RANDOM_SEED = 999
seed_everything(RANDOM_SEED)
g_cpu = torch.Generator().manual_seed(RANDOM_SEED)


@torch.no_grad()
def sample_model(unet, scheduler, c, scale, start_code):
    """Sample the model"""
    prev_noisy_sample = start_code
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(torch.cat([prev_noisy_sample] * 2), t, encoder_hidden_states=c).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
            prev_noisy_sample = scheduler.step(noise_pred, t, prev_noisy_sample).prev_sample
    return prev_noisy_sample

@torch.no_grad()
def inverse_model(x0, unet, scheduler, c):
    """Inverse the model to get noise"""
    next_noisy_sample = x0

    for t in torch.flip(scheduler_inversion.timesteps, [0]):
        with torch.no_grad():
            noise_pred = unet(torch.cat([next_noisy_sample] * 2), t, encoder_hidden_states=c).sample
            _, noise_pred_text = noise_pred.chunk(2)
            next_noisy_sample = scheduler.step(noise_pred_text, t, next_noisy_sample).prev_sample

    return next_noisy_sample


def decode_to_waveform(dec,vocoder,image_key='fbank'):
    if image_key == "fbank":
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav_reconstruction = vocoder_infer(dec, vocoder)
    elif image_key == "stft":
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav_reconstruction = wave_decoder(dec)
    return wav_reconstruction

def latents_to_audios(latents,vae,pipe,return_mel=False):
    # if scale:
    latents = 1 / vae.config.scaling_factor * latents
    mel_spec = vae.decode(latents).sample
    audio = pipe.mel_spectrogram_to_waveform(mel_spec)
    if return_mel == True:
        return audio, mel_spec
    else:
        audio = pipe.mel_spectrogram_to_waveform(mel_spec)
        return audio
    

def quick_sample(alpha,start_code1,start_code2,emb_1,emb_2,emb_prompt_1,emb_prompt_2,
    min_scale,max_scale,vae,pipe,sampling_steps = 15,use_unet_uncon =False,return_mel = False):
    # Interpolate embeddings
    new_emb = alpha * emb_1 + (1 - alpha) * emb_2
    new_prompt_emb = alpha * emb_prompt_1 + (1 - alpha)* emb_prompt_2
    scale = max_scale - np.abs(alpha - 0.5) * (max_scale-min_scale) * 2.0
    new_start_code = slerp((1-alpha), start_code1, start_code2)
    # Obtain x0
    x0_new = pipe.quick_backward_diffusion(
        latents=new_start_code,
        generated_prompt_embeds=new_prompt_emb,
        prompt_embeds = new_emb,
        attention_mask = attention_mask_2,
        guidance_scale=scale,
        num_inference_steps=sampling_steps,
        generator = g_cpu,
        use_unet_uncon= use_unet_uncon,
    )
    # Obtain audio waveforms
    if return_mel:
        y,mel = latents_to_audios(x0_new,vae,pipe,True)
        return y,mel
    else:
        y = latents_to_audios(x0_new,vae,pipe)
        return y

def normalize_columns_to_range(arr, range_min=-1, range_max=1):
    # Create a copy of the array to avoid modifying the original array
    arr_normalized = np.copy(arr)
    
    # Normalize each column separately
    for i in range(arr.shape[1]):
        col_min = np.min(arr[:, i])
        col_max = np.max(arr[:, i])
        
        if col_min == col_max:
            # If all values in the column are the same, set them to the mid-point of the desired range
            arr_normalized[:, i] = (range_max + range_min) / 2
        else:
            # Normalize to [0, 1]
            arr_normalized[:, i] = (arr[:, i] - col_min) / (col_max - col_min)
            
            # Scale to the desired range [range_min, range_max]
            arr_normalized[:, i] = arr_normalized[:, i] * (range_max - range_min) + range_min
        
    return arr_normalized



def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not -recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    # some implementation remove sin_theta_0, doesn't make much difference
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0 
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    if c:
        res = torch.from_numpy(v2).to(device)
    else:
        res = v2
    return res


class SPDPBinarySearch:
    def __init__(self, num_uniform_samples = 10,search_tolerance=1e-2, device='cuda:0',
                 sample_method=None,load_audio=latents_to_audios, slerp=True, 
                 quit_search_thresh=1e-8,sr = 16000,McAdam=True):
        self.sr = sr
        self.num_uniform_samples = num_uniform_samples
        self.search_tolerance = search_tolerance # tolerance for binary search error
        self.sample_method = sample_method # (embedding, guidance scale, start noise x_T), return a PIL image object
        self.load_audio = load_audio # preprocess of PIL image
        self.device = device
        self.slerp = slerp
        self.quit_search_thresh = quit_search_thresh
        self.McAdam = McAdam # set True, return McAdam timbre space

    def get_target_points(self,p0,p1):
        # Return N-2 interepolate points between p0 and p1
        points = []
        for i in range(1,self.num_uniform_samples-1):
            t = i / (self.num_uniform_samples - 1)
            p_i = (1 - t) * p0 + t * p1
            points.append(p_i)
        return np.array(points) 


    def timbre_space(self,y):
        # Compute perceptural points in the timbre space given audio y ans sr
        # McAdam returns the third dimension as spectral flux
        # Compute log attack time
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        onset_frames = np.nonzero(onset_env > 0.01)[0]
        if len(onset_frames) > 1:
            attack_time = librosa.frames_to_time(onset_frames[1], sr=self.sr) - librosa.frames_to_time(onset_frames[0], sr=self.sr)
            log_attack_time = np.log10(attack_time + 1e-5)  # Add a small value to avoid log(0)
        else:
            log_attack_time = -1 # cannot compute log attack time, set it as -1
        # Compute the spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr).mean()
        if self.McAdam == True:
            # Compute spectral flux
            spectral_flux = librosa.onset.onset_strength(y=y, sr=self.sr).mean()
            return np.array([log_attack_time,spectral_centroid,spectral_flux])
        else:
            # Compute spectral deviation
            S = np.abs(librosa.stft(y))
            sc = librosa.feature.spectral_centroid(S=S)
            spectral_deviation = np.mean(np.abs(S - sc), axis=0).mean()
            return np.array([log_attack_time,spectral_centroid,spectral_deviation])
            
    def compute_mel_spectrogram(self,y, n_components = 2, n_mels=128, n_fft=2048, hop_length=256):
        # Extract log Mel spectrogram given an audio sample
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db = np.squeeze(mel_spectrogram_db.T,axis=-1) #[time,mel bin]
        return mel_spectrogram_db

        
    def L2_tensor(self, audio_1, audio_2):
        # Input two audio waves, output L2 distance of mel spectrogram
        p1 = self.compute_mel_spectrogram(audio_1.cpu().detach().numpy())
        p2 = self.compute_mel_spectrogram(audio_2.cpu().detach().numpy())
        return np.linalg.norm((p1 - p2))

    def binary_search(self, current_audio, start_audio, end_audio, start_alpha, end_alpha, emb_1, emb_2,emb_prompt_1,emb_prompt_2,
                      min_scale, max_scale, start_code1, start_code2,p_target,vae,pipe,use_unet_uncon = False,sampling_steps = 20,
                      search_tolerance=5e-4,quit_search_thresh=1e-5, return_mel = True):
        # for each target intepolate point, use bineary search to find the closed alpha value
        start_alpha_temp = start_alpha
        end_alpha_temp = end_alpha
        y_mid, mel_mid= quick_sample(start_alpha_temp,start_code1,start_code2,emb_1,emb_2,emb_prompt_1,emb_prompt_2,min_scale,max_scale,vae,pipe,sampling_steps = sampling_steps, use_unet_uncon = use_unet_uncon, return_mel = return_mel)
        distance_a0 = self.L2_tensor(y_mid, start_audio)
        distance_a1 = self.L2_tensor(y_mid, end_audio)
        total_distance = distance_a0 + distance_a1
        # Obtain current SPDP point
        p_mid = np.array([distance_a1 / total_distance, distance_a0 / total_distance])
        while np.abs(start_alpha_temp - end_alpha) > quit_search_thresh: 
            alpha_mid = (start_alpha_temp + end_alpha) / 2
            # mid spect, mid audio
            y_mid, mel_mid= quick_sample(alpha_mid,start_code1,start_code2,emb_1,emb_2,emb_prompt_1,emb_prompt_2,min_scale,max_scale,vae,pipe,sampling_steps = sampling_steps, use_unet_uncon = use_unet_uncon, return_mel = return_mel)
            distance_a0 = self.L2_tensor(y_mid, start_audio)
            distance_a1 = self.L2_tensor(y_mid, end_audio)
            total_distance = distance_a0 + distance_a1
            # Obtain intermediate proportion point
            p_mid = np.array([distance_a1 / total_distance, distance_a0 / total_distance])
            if np.linalg.norm((p_mid-p_target)) < search_tolerance:
                return (start_alpha_temp + end_alpha) / 2, p_mid
            if p_mid[0] > p_target[0][0]:
                end_alpha = (start_alpha_temp + end_alpha) / 2
            else:
                start_alpha_temp = (start_alpha_temp + end_alpha) / 2
        return (start_alpha_temp + end_alpha) / 2, p_mid

    
    def search(self,start_audio, end_audio, emb_1, emb_2,emb_prompt_1,emb_prompt_2, min_scale, 
        max_scale, start_code1, start_code2,target_SPDP,vae,pipe,use_unet_uncon = False, 
        sampling_steps = 20,start_alpha=0, end_alpha=1,
        search_tolerance=1e-2,quit_search_thresh=1e-8,num_uniform_samples=10):
        alpha_list = [0]
        end_alpha = end_alpha
        current_alpha = start_alpha
        current_audio = start_audio
        print("Starting binary search ...")
        with tqdm(total=self.num_uniform_samples-2) as progress_bar:
            for idx in range(len(target_SPDP)):
                target_point = target_SPDP[idx].reshape(1,-1)
                current_alpha, current_audio = self.binary_search(current_audio, start_audio,end_audio,current_alpha, end_alpha, 
                           emb_1, emb_2, emb_prompt_1,emb_prompt_2,min_scale,
                           max_scale, start_code1, start_code2,target_point, vae, pipe,use_unet_uncon = use_unet_uncon, sampling_steps = sampling_steps)
                alpha_list.append(current_alpha)
                progress_bar.update(1)
        if np.abs(alpha_list[-1] - 1) > quit_search_thresh:
            alpha_list = alpha_list + [1]
        return alpha_list

from diffusers import AutoencoderKL,AudioLDM2UNet2DConditionModel,AudioLDM2ProjectionModel
from diffusers import DDIMScheduler, DDIMInverseScheduler
from transformers import RobertaTokenizer,ClapFeatureExtractor,GPT2Model,ClapModel,T5Tokenizer,T5EncoderModel,SpeechT5HifiGan,ClapFeatureExtractor, GPT2Model

def process_audio_pair(input_dir, a_prev, a_next):
    wav_ref = cdpam.load_audio(os.path.join(input_dir, a_prev))
    wav_out = cdpam.load_audio(os.path.join(input_dir, a_next))
    return wav_ref, wav_out


# Function to split an audio into N clips
def split_audio(audio, sr, num_clips):
    clip_length = len(audio) // num_clips
    clips = [audio[i*clip_length:(i+1)*clip_length] for i in range(num_clips)]
    return clips

# Function to split an audio into N clips
def split_latent(x0, num_clips):
    clip_length = x0.shape[2] // num_clips
    clips = [x0[:,:,i*clip_length:(i+1)*clip_length,:] for i in range(num_clips)]
    return clips

# Function to concatenate the i-th clip from each audio
def concatenate_clips(clips_list):
    concatenated_audio = np.concatenate([clips.reshape(1,-1) for clips in clips_list])
    return concatenated_audio


def main(input_sample_1,input_sample_2,prompt1, prompt2,subfolder_name,dir,sampling_steps=100,num_uniform_samples=10,min_scale=1.5,max_scale=3.5):
    # This is a manuscript for batch run music morphing
    import os
    # if not os.path.exists(f'./{dir}/{subfolder_name}/'):
    os.makedirs(f'./{dir}/{subfolder_name}/',exist_ok=True)

    inversion_steps = 100
    if sampling_steps is None:
        sampling_steps = 100
    # Init preprocessor
    
    # Read config
    config_yaml = f"config/autoencoder/16k_64.yaml"
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)

    # Obtain preprocessor
    preprocessor = Preprocessor(config_yaml)
    dtype = torch.float32
    model_id = "cvssp/audioldm2"

    # Init model
    print(f'load models from path: {model_id}')

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = RobertaTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=dtype)
    tokenizer_2 = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer_2", torch_dtype=dtype)

    text_encoder = ClapModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)

    # 3. The UNet model for generating the latents.
    unet = AudioLDM2UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    unet2 = AudioLDM2UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

    # 4. Scheduler for training images, eta is by default 0
    scheduler_training = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=dtype)

    # 5. Scheduler for sampling images, eta is by default 0
    scheduler_sampling = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=dtype)

    # 6. Scheduler for inversion into noise, eta is by default 0
    scheduler_inversion = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=dtype)

    # 7. Get vocoder
    vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm2",subfolder="vocoder")

    # 8. Get feature extractor
    feature_extractor = ClapFeatureExtractor.from_pretrained("cvssp/audioldm2",subfolder="feature_extractor", torch_dtype=dtype)

    # 9. Get language model
    language_model = GPT2Model.from_pretrained("cvssp/audioldm2",subfolder="language_model", torch_dtype=dtype)

    # 10. Get projection model
    projection_model= AudioLDM2ProjectionModel.from_pretrained("cvssp/audioldm2",subfolder="projection_model", torch_dtype=dtype)

    # 11. Get overall audioldm pipeline 
    from pipeline_audioldm2_modified import AudioLDM2Pipeline
    pipe = AudioLDM2Pipeline(vae,text_encoder,text_encoder_2,projection_model,language_model,tokenizer,tokenizer_2,
                         feature_extractor,unet,scheduler_sampling,scheduler_inversion,vocoder,None)


    unet = unet.to(device)
    unet2 = unet2.to(device)
    vocoder = vocoder.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    language_model = language_model.to(device)
    projection_model = projection_model.to(device)
    # pip = pipe.to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    unet2.requires_grad_(False)
    language_model.requires_grad_(False)
    projection_model.requires_grad_(False)
    vocoder.requires_grad_(False)

    log_mel_spec1, stft1, waveform1, random_start1  = preprocessor.read_audio_file(filename=input_sample_1)
    log_mel_spec1 = log_mel_spec1.float().to(device).unsqueeze(0).unsqueeze(0)
    init_latent_1 = vae.config.scaling_factor * vae.encode(log_mel_spec1).latent_dist.sample(g_cpu)
    init_latent_1 = init_latent_1.to(dtype)


    # Audio Sample 2
    log_mel_spec2, stft2, waveform2, random_start2  = preprocessor.read_audio_file(filename=input_sample_2)
    log_mel_spec2 = log_mel_spec2.float().to(device).unsqueeze(0).unsqueeze(0)
    init_latent_2 = vae.config.scaling_factor * vae.encode(log_mel_spec2).latent_dist.sample(g_cpu)
    init_latent_2 = init_latent_2.to(dtype)

    negative_prompt = 'low quality'
    global attention_mask_2
    global attention_mask_1

    with torch.no_grad():
        prompt_embeds_1, attention_mask_1,genearted_prompt_embeds_1 = pipe.encode_prompt(prompt1,device,1,True,negative_prompt = negative_prompt)
        prompt_embeds_2, attention_mask_2,genearted_prompt_embeds_2 = pipe.encode_prompt(prompt2,device,1,True,negative_prompt = negative_prompt)

    emb_1 = prompt_embeds_1.clone()
    emb_prompt_1 = genearted_prompt_embeds_1.clone()
    emb_2 = prompt_embeds_2.clone()
    emb_prompt_2 = genearted_prompt_embeds_2.clone()

    
    # Text inversion    
    emb_1.requires_grad = True
    emb_2.requires_grad = True
    attention_mask_1.requires_grad=False
    attention_mask_2.requires_grad=False
    emb_prompt_1.require_grad=True
    emb_prompt_2.require_grad=True
    

    lr = 2e-3
    it = 2500
    start_code = randn_tensor(init_latent_1.shape, generator=g_cpu, dtype=dtype)
    start_code = start_code * scheduler_sampling.init_noise_sigma
    start_code = start_code.to(device)
    num_train_timesteps = len(scheduler_training.betas)
    scheduler_training.set_timesteps(num_train_timesteps)
    scheduler_sampling.set_timesteps(sampling_steps)
    scheduler_inversion.set_timesteps(inversion_steps)

    opt = torch.optim.Adam([emb_1,emb_prompt_1], lr=lr)

    # opt = torch.optim.Adam([emb_1], lr=lr)
    criteria = torch.nn.MSELoss()
    history = []
    pbar = tqdm(range(it))
    for i in pbar:
        opt.zero_grad()
        noise = randn_tensor(init_latent_1.shape, generator=g_cpu, dtype=dtype).to(device)
        t_enc = torch.randint(num_train_timesteps, (1,), device=device)
        z = scheduler_training.add_noise(init_latent_1, noise, t_enc)
        pred_noise = predict_noise0_diffuser(unet, z, emb_prompt_1, emb_1, attention_mask_1, t_enc, dtype = dtype, guidance_scale=1 ,scheduler=scheduler_training)
        loss = criteria(pred_noise, noise) # float32
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
        

    torch.cuda.empty_cache()

    opt = torch.optim.Adam([emb_2,emb_prompt_2], lr=lr)
    criteria = torch.nn.MSELoss()
    history = []
    pbar = tqdm(range(it))
    for i in pbar:
        opt.zero_grad()
        noise = randn_tensor(init_latent_2.shape, generator=g_cpu, dtype=dtype).to(device)
        t_enc = torch.randint(num_train_timesteps, (1,), device=device)
        z = scheduler_training.add_noise(init_latent_2, noise, t_enc)
        pred_noise = predict_noise0_diffuser(unet, z, emb_prompt_2, emb_2, attention_mask_2, t_enc, dtype = dtype, guidance_scale=1 ,scheduler=scheduler_training)
        loss = criteria(pred_noise, noise) # float32
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    # LoRA fine-tune
    LORA_RANK = 4
    unet = unet.eval()
    unet2 = unet2.eval()
    
    torch.cuda.empty_cache()
    lr = 1e-3
    unet_lora, unet_lora_layers = extract_lora_diffusers(unet, device, rank = LORA_RANK)
    lora_params = list(unet_lora_layers.parameters())
    for param in unet_lora.parameters():
        param.requires_grad = True
    unet_lora.to(device)
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lr}], lr=lr)
    print(f'number of trainable parameters of LoRA model in optimizer: {sum(p.numel() for p in lora_params if p.requires_grad)}')

    emb_1.requires_grad = False
    emb_prompt_1.requires_grad = False
    emb_2.requires_grad = False
    emb_prompt_2.requires_grad = False

    unet_lora.train()
    it = 150
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(it))

    for i in pbar:
        opt.zero_grad()
        # noise = torch.randn_like(init_latent_1)
        noise = randn_tensor(init_latent_1.shape, generator=g_cpu, dtype=dtype).to(device)
        # start_code = start_code * scheduler_sampling.init_noise_sigma
        t_enc = torch.randint(num_train_timesteps, (1,), device=device)
        z = scheduler_training.add_noise(init_latent_1, noise, t_enc)
        
        pred_noise = predict_noise0_diffuser(unet_lora, z, emb_prompt_1, emb_1, attention_mask_1, t_enc, 
                                             dtype = dtype, guidance_scale=1,
                                             scheduler=scheduler_training, cross_attention_kwargs = {'scale': 1})
        
        loss = criteria(pred_noise, noise)
        loss.backward()
        
        noise = randn_tensor(init_latent_2.shape, generator=g_cpu, dtype=dtype).to(device)
        t_enc = torch.randint(num_train_timesteps, (1,), device=device)
        z = scheduler_training.add_noise(init_latent_2, noise, t_enc)
        pred_noise = predict_noise0_diffuser(unet_lora, z, emb_prompt_2, emb_2, attention_mask_2, t_enc, 
                                             dtype = dtype, guidance_scale=1,
                                             scheduler=scheduler_training, cross_attention_kwargs = {'scale': 1})
        loss = criteria(pred_noise, noise)
        loss.backward()
        
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    torch.cuda.empty_cache()
    LORA_RANK_uncond = 2
    lr = 1e-3
    unet_lora_uncond, unet_lora_layers_uncond = extract_lora_diffusers(unet2, device, rank = LORA_RANK_uncond)
    lora_params_uncond = list(unet_lora_layers_uncond.parameters())
    unet_lora_uncond.to(device)
    opt = torch.optim.AdamW([{"params": lora_params_uncond, "lr": lr}], lr=lr)
    print(f'number of trainable parameters of LoRA model in optimizer: {sum(p.numel() for p in lora_params_uncond if p.requires_grad)}')

    for param in unet_lora_uncond.parameters():
        param.requires_grad = True

    prompt_embeds_uncond, attention_mask_uncond,genearted_prompt_embeds_uncond = pipe.encode_prompt(["low quality"],device,1,True,negative_prompt = [""])

    emb_1.requires_grad = False
    emb_prompt_1.requires_grad = False
    emb_2.requires_grad = False
    emb_prompt_2.requires_grad = False

    unet_lora_uncond.train()
    it = 15
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(it))
    for i in pbar:
        opt.zero_grad()
        # noise = torch.randn_like(init_latent_1)
        noise = randn_tensor(init_latent_1.shape, generator=g_cpu, dtype=dtype).to(device)
        t_enc = torch.randint(num_train_timesteps, (1,), device=device)
        z = scheduler_training.add_noise(init_latent_1, noise, t_enc)
        pred_noise = predict_noise0_diffuser(unet_lora_uncond, z, genearted_prompt_embeds_uncond, prompt_embeds_uncond, attention_mask_uncond, t_enc, 
                                             dtype = dtype, guidance_scale=1 ,
                                             scheduler=scheduler_training,cross_attention_kwargs = {'scale': 1})

        loss = criteria(pred_noise, noise)
        loss.backward()
        
        # prior preserve embedding - joint optimzation with generate image (bird spread wings)
        # noise = torch.randn_like(init_latent_2)
        noise = randn_tensor(init_latent_2.shape, generator=g_cpu, dtype=dtype).to(device)
        t_enc = torch.randint(num_train_timesteps, (1,), device=device)
        #z = model.q_sample(init_latent_2, t_enc, noise=noise)
        z = scheduler_training.add_noise(init_latent_2, noise, t_enc)
        #pred_noise = model.apply_model(z, t_enc, emb_2)
        pred_noise = predict_noise0_diffuser(unet_lora_uncond, z, genearted_prompt_embeds_uncond, prompt_embeds_uncond, attention_mask_uncond, t_enc, 
                                             dtype = dtype, guidance_scale=1 ,
                                             scheduler=scheduler_training, cross_attention_kwargs = {'scale': 1})
        loss = criteria(pred_noise, noise)
        loss.backward()
        
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    unet_lora_uncond = unet_lora_uncond.eval()
    unet_lora = unet_lora.eval()

    os.makedirs(f'./{dir}/{subfolder_name}/con_lora/',exist_ok=True)
    os.makedirs(f'./{dir}/{subfolder_name}/uncon_lora/',exist_ok=True)   
    unet_lora.save_attn_procs(save_directory=f'./{dir}/{subfolder_name}/con_lora/')
    unet_lora_uncond.save_attn_procs(save_directory=f'./{dir}/{subfolder_name}/uncon_lora/')


    use_unet_uncon = True
    if use_unet_uncon == True:
        from pipeline_audioldm2_modified import AudioLDM2Pipeline
        pipe = AudioLDM2Pipeline(vae,text_encoder,text_encoder_2,projection_model,language_model,tokenizer,tokenizer_2,
                                 feature_extractor,unet_lora,scheduler_sampling,scheduler_inversion,vocoder,unet_uncond=unet_lora_uncond)

    # Start init code
    start_code1 = pipe.forward_diffusion(
        latents=init_latent_1,
        generated_prompt_embeds=emb_prompt_1,
        prompt_embeds = emb_1,
        attention_mask = attention_mask_1,
        guidance_scale=1,
        num_inference_steps=inversion_steps,
        generator = g_cpu,
    )
    #sanity check for ddim inversion, should be close to 1
    print('variance of inverted xT:', start_code1.var())

    x0 = pipe.backward_diffusion(
        latents=start_code1,
        generated_prompt_embeds=emb_prompt_1,
        prompt_embeds = emb_1,
        attention_mask = attention_mask_1,
        guidance_scale=min_scale,
        num_inference_steps=sampling_steps,
        generator = g_cpu,
        use_unet_uncon=use_unet_uncon,
    )
    start_audio,start_mel = latents_to_audios(x0,vae,pipe,True)

    start_code2 = pipe.forward_diffusion(
        latents= init_latent_2,
        generated_prompt_embeds=emb_prompt_2,
        prompt_embeds = emb_2,
        attention_mask = attention_mask_2,
        guidance_scale=1,
        num_inference_steps=inversion_steps,
        generator = g_cpu,
    )
    x0 = pipe.backward_diffusion(
        latents= start_code2,
        generated_prompt_embeds=emb_prompt_2,
        prompt_embeds = emb_2,
        attention_mask = attention_mask_2,
        guidance_scale=min_scale,
        num_inference_steps=sampling_steps,
        generator = g_cpu,
        use_unet_uncon=use_unet_uncon,
    )
    end_audio,end_mel = latents_to_audios(x0,vae,pipe,True)


    sampling_alpha = SPDPBinarySearch(num_uniform_samples = num_uniform_samples, search_tolerance=1e-2,
                                       sample_method=quick_sample, load_audio=latents_to_audios)
    # Obtain num_uniform_samples target p points
    p_targets = sampling_alpha.get_target_points(np.array([0,1]),np.array([1,0]))
    alpha_list = sampling_alpha.search(start_audio, end_audio, emb_1, emb_2, emb_prompt_1,emb_prompt_2,min_scale, max_scale,start_code1,start_code2,
                                    target_SPDP = p_targets,vae = vae,pipe = pipe, use_unet_uncon = True, sampling_steps = sampling_steps, start_alpha=0, end_alpha=1, num_uniform_samples=num_uniform_samples)

    with open(f"{dir}/{subfolder_name}/alpha_list.txt", 'w') as file:
        file.write(str(alpha_list))

    all_timbre_points = []
    all_perceptual_points = []
    mel_sequence_xT_slerp = []
    audio_sequence_xT_slerp = []
    x0_list = []

    os.makedirs(f"{dir}/{subfolder_name}/audios", exist_ok=True)
    for idx,alpha_i in enumerate(alpha_list):
        alpha = alpha_i
        emb = alpha * emb_1 + (1 - alpha) * emb_2
        prompt_emb = alpha * emb_prompt_1 + (1 - alpha) * emb_prompt_2
        # prompt_emb = slerp((1-alpha), emb_prompt_1, emb_prompt_2)

        scale = max_scale - np.abs(alpha - 0.5) * (max_scale-min_scale) * 2.0
        new_start_code = slerp((1-alpha), start_code1, start_code2)
        
        x0_new = pipe.backward_diffusion(
        latents=new_start_code,
        generated_prompt_embeds=prompt_emb,
        prompt_embeds = emb,
        attention_mask = attention_mask_1,
        guidance_scale=scale,
        num_inference_steps=sampling_steps,
        generator = g_cpu,
        use_unet_uncon= use_unet_uncon,
        )

        x0_list.append(x0_new)
        audio,mel = latents_to_audios(x0_new,vae,pipe,True)
        mel_sequence_xT_slerp.append(mel[0]) #[1,1024,64]
        audio_sequence_xT_slerp.append(audio[0])
        print("Alpha value:",alpha)
        # Save audios
        write(f"{dir}/{subfolder_name}/audios/{alpha_i}.wav",16000, audio[0].cpu().detach().numpy())
    # Dynamic morphing
    num_audios = len(audio_sequence_xT_slerp)

    # Initialize lists to store audio data and sample rate
    audio_data = []

    # Load audio files and split into clips
    all_clips = []
    for x_i in x0_list:
        
        x_i = x_i.cpu().detach().numpy()
    
        clips = split_latent(x_i, num_audios)
        all_clips.append(clips)


    # Concatenate the i-th clip from each audio
    concate_list = []
    for i in range(num_audios):
        concate_list.append(all_clips[i][i])
    final_x0 = concatenate_clips(concate_list)
    # Decode back to waveform
    final_audio = latents_to_audios(torch.Tensor(final_x0).cuda(),vae,pipe,False)
    write(f"{dir}/{subfolder_name}/dynamic_morph.wav",16000, final_audio[0].cpu().detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_size", type=int, default = 10, required=True)
    parser.add_argument("--sampling_steps", type=int, default = 100)
    parser.add_argument("--max_scale", type=float, default = 4)
    parser.add_argument("--min_scale", type=float, default = 1.5)
    args = parser.parse_args()


    device = "cuda:0"
    dir = "music"
    input_dir = f"./audio"
    meta_path = f"./audio/meta.txt"

    prompt1 = 'a audio clip of music composition.' 
    prompt2 = 'a audio clip of music composition.' 

    # min_scale=1.5
    # max_scale=4
    # num_uniform_samples = 15
    min_scale=args.min_scale
    max_scale=args.max_scale
    num_uniform_samples = args.N_size
    sampling_steps = args.sampling_steps

    with open(meta_path, 'r') as f:
        pair_list = [line.strip() for line in f]

    # Iterate execute meta file
    for pair_str in pair_list:
    
        pair = pair_str.split(", ")
        subfolder_name = pair[0]+"_"+pair[1]

        input_sample_1 = os.path.join(f"./audio",pair[0])
        input_sample_2 =  os.path.join(f"./audio",pair[1])
        main(input_sample_1,input_sample_2,prompt1,prompt2,subfolder_name,dir,sampling_steps = sampling_steps,num_uniform_samples=num_uniform_samples,min_scale=min_scale,max_scale=max_scale)
        torch.cuda.empty_cache()