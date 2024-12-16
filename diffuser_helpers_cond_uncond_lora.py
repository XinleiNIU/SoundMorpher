from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)


from diffusers.loaders import AttnProcsLayers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler 
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import pdb
def extract_lora_diffusers(unet, device, dtype=None, rank=4):
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers, default rank is 4
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None
        if name.endswith("attn1.processor"):
            cross_attention_dim = None
        elif name.endswith("attn2.processor"):
            name_list = name.split(".")
            idx = name_list.index('attentions')+1
    
            if int(name_list[idx])%3 == 0:
                cross_attention_dim = None 
            elif int(name_list[idx])%3 == 1:
                cross_attention_dim = 768
            elif int(name_list[idx])%3 == 2:
                cross_attention_dim = 1024
            else:
                cross_attention_dim = None
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor #LoRAAttnProcessor
        unet_lora_attn_procs[name] = lora_attn_processor_class(hidden_size=hidden_size,cross_attention_dim=cross_attention_dim, rank=rank).to(device)
        # print("complete one loop")
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # self.unet.requires_grad_(True)
    unet.requires_grad_(False)
    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)
    ### end lora
    if not dtype:
        unet = unet.to(dtype)
        unet_lora_layers = unet_lora_layers.to(dtype)
    return unet, unet_lora_layers

def predict_noise0_diffuser(unet, noisy_latents, generated_prompt_embeds, prompt_embeds, 
    attention_mask, t, dtype=torch.float32, guidance_scale=7.5, cross_attention_kwargs={}, 
    scheduler=None, lora_v=False, half_inference=False, mask_uncon = False):
    batch_size = noisy_latents.shape[0]
    # for both conditioned & unconditioned generation
    
    if guidance_scale == 1.:
        latent_model_input = noisy_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        #noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
        #                  cross_attention_kwargs=cross_attention_kwargs).sample
        noise_pred = unet(latent_model_input, 
            t, 
            encoder_hidden_states=generated_prompt_embeds[batch_size:],
            encoder_hidden_states_1=prompt_embeds[batch_size:],
            encoder_attention_mask_1=attention_mask[batch_size:],
            cross_attention_kwargs=cross_attention_kwargs).sample
        noise_pred_text = noise_pred#.chunk(2)
        return noise_pred_text
    else:
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=generated_prompt_embeds,
            encoder_hidden_states_1=prompt_embeds,
            encoder_attention_mask_1=attention_mask,
            return_dict=False,
            cross_attention_kwargs=cross_attention_kwargs).sample


        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if mask_uncon:
            noise_pred_text.register_hook(lambda grad: grad * torch.zeros_like(grad).to(dtype))
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.to(dtype)
    return noise_pred
