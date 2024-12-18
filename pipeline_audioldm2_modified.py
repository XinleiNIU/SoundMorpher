# Copyright 2024 CVSSP, ByteDance and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn.functional as F
import numpy as np
import torch
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
    VitsModel,
    VitsTokenizer,
)

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler,DDIMInverseScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import Attention
# from ...utils import (
#     is_accelerate_available,
#     is_accelerate_version,
#     is_librosa_available,
#     logging,
#     replace_example_docstring,
# )
from diffusers.utils.torch_utils import randn_tensor
try:
    from diffusers.pipeline_utils import DiffusionPipeline,AudioPipelineOutput
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline,AudioPipelineOutput
from diffusers import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel
from functools import partial

# if is_librosa_available():
import librosa
from diffusers.utils import logging,replace_example_docstring

from utils.utils import wav_to_fbank, TacotronSTFT
import torchaudio
import pdb
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


CURRENT_LOSS = None
CURRENT_TIMESTEP = None


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
        ```
        ```
        #Using AudioLDM2 for Text To Speech
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "anhnct/audioldm2_gigaspeech"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "A female reporter is speaking"
        >>> transcript = "wish you have a good day"

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     transcription=transcript,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=2,
        ...     generator=generator,
        ...     max_new_tokens=512,          #Must set max_new_tokens equa to 512 for TTS
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("tts.wav", rate=16000, data=audio[0])
        ```
"""


def prepare_inputs_for_generation(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }


class AudioLDM2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using AudioLDM2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.ClapModel`]):
            First frozen text-encoder. AudioLDM2 uses the joint audio-text embedding model
            [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection),
            specifically the [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant. The
            text branch is used to encode the text prompt to a prompt embedding. The full audio-text model is used to
            rank generated waveforms against the text prompt by computing similarity scores.
        text_encoder_2 ([`~transformers.T5EncoderModel`, `~transformers.VitsModel`]):
            Second frozen text-encoder. AudioLDM2 uses the encoder of
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) variant. Second frozen text-encoder use
            for TTS. AudioLDM2 uses the encoder of
            [Vits](https://huggingface.co/docs/transformers/model_doc/vits#transformers.VitsModel).
        projection_model ([`AudioLDM2ProjectionModel`]):
            A trained model used to linearly project the hidden-states from the first and second text encoder models
            and insert learned SOS and EOS token embeddings. The projected hidden-states from the two text encoders are
            concatenated to give the input to the language model. A Learned Position Embedding for the Vits
            hidden-states
        language_model ([`~transformers.GPT2Model`]):
            An auto-regressive language model used to generate a sequence of hidden-states conditioned on the projected
            outputs from the two text encoders.
        tokenizer ([`~transformers.RobertaTokenizer`]):
            Tokenizer to tokenize text for the first frozen text-encoder.
        tokenizer_2 ([`~transformers.T5Tokenizer`, `~transformers.VitsTokenizer`]):
            Tokenizer to tokenize text for the second frozen text-encoder.
        feature_extractor ([`~transformers.ClapFeatureExtractor`]):
            Feature extractor to pre-process generated audio waveforms to log-mel spectrograms for automatic scoring.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            Vocoder of class `SpeechT5HifiGan` to convert the mel-spectrogram latents to the final audio waveform.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapModel,
        text_encoder_2: Union[T5EncoderModel, VitsModel],
        projection_model: AudioLDM2ProjectionModel,
        language_model: GPT2Model,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        tokenizer_2: Union[T5Tokenizer, T5TokenizerFast, VitsTokenizer],
        feature_extractor: ClapFeatureExtractor,
        unet: AudioLDM2UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        scheduler_inversion: KarrasDiffusionSchedulers,
        # scheduler: DDIMScheduler, # KarrasDiffusionSchedulers,
        # scheduler_inversion: DDIMInverseScheduler,# KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
        unet_uncond: AudioLDM2UNet2DConditionModel = None
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            projection_model=projection_model,
            language_model=language_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.inverse_scheduler = scheduler_inversion
        self.unet_uncond = unet_uncond

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = [
            self.text_encoder.text_model,
            self.text_encoder.text_projection,
            self.text_encoder_2,
            self.projection_model,
            self.language_model,
            self.unet,
            self.vae,
            self.vocoder,
            self.text_encoder,
        ]

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def generate_language_model(
        self,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 8,
        **model_kwargs,
    ):
        """

        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = self.language_model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

    def preprocessing(
        self,
        audio_path):

        # 3. Preprocess audio
        audio, sample_rate = torchaudio.load(audio_path)
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        audio_length_in_s = audio_16k.shape[1] / 16000
        height = int(audio_length_in_s / vocoder_upsample_factor)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        fn_STFT = TacotronSTFT(
            1024,  # config["preprocessing"]["stft"]["filter_length"],
            160,  # config["preprocessing"]["stft"]["hop_length"],
            1024,  # config["preprocessing"]["stft"]["win_length"],
            64,  # config["preprocessing"]["mel"]["n_mel_channels"],
            16000,  # config["preprocessing"]["audio"]["sampling_rate"],
            0,  # config["preprocessing"]["mel"]["mel_fmin"],
            8000,  # config["preprocessing"]["mel"]["mel_fmax"],
        )

        mel, _, _ = wav_to_fbank(
            audio_path, target_length=int(audio_length_in_s * 102.4), fn_STFT=fn_STFT
        )
        # mel = mel.unsqueeze(0).unsqueeze(0)
        return mel, audio_length_in_s


    def encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        transcription=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generated_prompt_embeds: Optional[torch.Tensor] = None,
        negative_generated_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            transcription (`str` or `List[str]`):
                transcription of text to speech
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, text embeddings will be computed from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            generated_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                The number of new tokens to generate with the GPT2 language model.
        Returns:
            prompt_embeds (`torch.Tensor`):
                Text embeddings from the Flan T5 model.
            attention_mask (`torch.LongTensor`):
                Attention mask to be applied to the `prompt_embeds`.
            generated_prompt_embeds (`torch.Tensor`):
                Text embeddings generated from the GPT2 langauge model.

        Example:

        ```python
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # Get text embedding vectors
        >>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        ...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
        ...     device="cuda",
        ...     do_classifier_free_guidance=True,
        ... )

        >>> # Pass text embeddings to pipeline for text-conditional audio generation
        >>> audio = pipe(
        ...     prompt_embeds=prompt_embeds,
        ...     attention_mask=attention_mask,
        ...     generated_prompt_embeds=generated_prompt_embeds,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ... ).audios[0]

        >>> # save generated audio sample
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```"""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        is_vits_text_encoder = isinstance(self.text_encoder_2, VitsModel)

        if is_vits_text_encoder:
            text_encoders = [self.text_encoder, self.text_encoder_2.text_encoder]
        else:
            text_encoders = [self.text_encoder, self.text_encoder_2]

        if prompt_embeds is None:
            prompt_embeds_list = []
            attention_mask_list = []

            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                use_prompt = isinstance(
                    tokenizer, (RobertaTokenizer, RobertaTokenizerFast, T5Tokenizer, T5TokenizerFast)
                )
                text_inputs = tokenizer(
                    prompt if use_prompt else transcription,
                    padding="max_length"
                    if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer))
                    else True,
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                attention_mask = text_inputs.attention_mask
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                        f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                text_input_ids = text_input_ids.to(device)
                attention_mask = attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    prompt_embeds = text_encoder.get_text_features(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    prompt_embeds = prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    attention_mask = attention_mask.new_ones((batch_size, 1))
                elif is_vits_text_encoder:
                    # Add end_token_id and attention mask in the end of sequence phonemes
                    for text_input_id, text_attention_mask in zip(text_input_ids, attention_mask):
                        for idx, phoneme_id in enumerate(text_input_id):
                            if phoneme_id == 0:
                                text_input_id[idx] = 182
                                text_attention_mask[idx] = 1
                                break
                    prompt_embeds = text_encoder(
                        text_input_ids, attention_mask=attention_mask, padding_mask=attention_mask.unsqueeze(-1)
                    )
                    prompt_embeds = prompt_embeds[0]
                else:
                    prompt_embeds = text_encoder(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    prompt_embeds = prompt_embeds[0]

                prompt_embeds_list.append(prompt_embeds)
                attention_mask_list.append(attention_mask)

            # End for loop here and obtain GPT-2 embedding
            projection_output = self.projection_model(
                hidden_states=prompt_embeds_list[0],
                hidden_states_1=prompt_embeds_list[1],
                attention_mask=attention_mask_list[0],
                attention_mask_1=attention_mask_list[1],
            )
            projected_prompt_embeds = projection_output.hidden_states
            projected_attention_mask = projection_output.attention_mask

            generated_prompt_embeds = self.generate_language_model(
                projected_prompt_embeds,
                attention_mask=projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        ## Obtain T5 embeddings
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        attention_mask = (
            attention_mask.to(device=device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.language_model.dtype, device=device)

        bs_embed, seq_len, hidden_size = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len, hidden_size)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_waveforms_per_prompt, seq_len)

        bs_embed, seq_len, hidden_size = generated_prompt_embeds.shape
        # duplicate generated embeddings for each generation per prompt, using mps friendly method
        generated_prompt_embeds = generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        generated_prompt_embeds = generated_prompt_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )


        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            negative_attention_mask_list = []
            max_length = prompt_embeds.shape[1]
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=tokenizer.model_max_length
                    if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer))
                    else max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                uncond_input_ids = uncond_input.input_ids.to(device)
                negative_attention_mask = uncond_input.attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    negative_prompt_embeds = text_encoder.get_text_features(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    negative_prompt_embeds = negative_prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    negative_attention_mask = negative_attention_mask.new_ones((batch_size, 1))
                elif is_vits_text_encoder:
                    negative_prompt_embeds = torch.zeros(
                        batch_size,
                        tokenizer.model_max_length,
                        text_encoder.config.hidden_size,
                    ).to(dtype=self.text_encoder_2.dtype, device=device)
                    negative_attention_mask = torch.zeros(batch_size, tokenizer.model_max_length).to(
                        dtype=self.text_encoder_2.dtype, device=device
                    )
                else:
                    negative_prompt_embeds = text_encoder(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    negative_prompt_embeds = negative_prompt_embeds[0]

                negative_prompt_embeds_list.append(negative_prompt_embeds)
                negative_attention_mask_list.append(negative_attention_mask)

            projection_output = self.projection_model(
                hidden_states=negative_prompt_embeds_list[0],
                hidden_states_1=negative_prompt_embeds_list[1],
                attention_mask=negative_attention_mask_list[0],
                attention_mask_1=negative_attention_mask_list[1],
            )
            negative_projected_prompt_embeds = projection_output.hidden_states
            negative_projected_attention_mask = projection_output.attention_mask

            negative_generated_prompt_embeds = self.generate_language_model(
                negative_projected_prompt_embeds,
                attention_mask=negative_projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_attention_mask = (
                negative_attention_mask.to(device=device)
                if negative_attention_mask is not None
                else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
            )
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
                dtype=self.language_model.dtype, device=device
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len, -1)

            # duplicate unconditional attention mask for each generation per prompt
            negative_attention_mask = negative_attention_mask.repeat(1, num_waveforms_per_prompt)
            negative_attention_mask = negative_attention_mask.view(batch_size * num_waveforms_per_prompt, seq_len)

            # duplicate unconditional generated embeddings for each generation per prompt
            seq_len = negative_generated_prompt_embeds.shape[1]
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.view(
                batch_size * num_waveforms_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([negative_attention_mask, attention_mask])
            generated_prompt_embeds = torch.cat([negative_generated_prompt_embeds, generated_prompt_embeds])

        return prompt_embeds, attention_mask, generated_prompt_embeds

    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.mel_spectrogram_to_waveform
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype):
        if not is_librosa_available():
            logger.info(
                "Automatic scoring of the generated audio waveforms against the input prompt text requires the "
                "`librosa` package to resample the generated waveforms. Returning the audios in the order they were "
                "generated. To enable automatic scoring, install `librosa` with: `pip install librosa`."
            )
            return audio
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        resampled_audio = librosa.resample(
            audio.numpy(), orig_sr=self.vocoder.config.sampling_rate, target_sr=self.feature_extractor.sampling_rate
        )
        inputs["input_features"] = self.feature_extractor(
            list(resampled_audio), return_tensors="pt", sampling_rate=self.feature_extractor.sampling_rate
        ).input_features.type(dtype)
        inputs = inputs.to(device)

        # compute the audio-text similarity score using the CLAP model
        logits_per_text = self.text_encoder(**inputs).logits_per_text
        # sort by the highest matching generations per prompt
        indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
        audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
        return audio

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        transcription=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        generated_prompt_embeds=None,
        negative_generated_prompt_embeds=None,
        attention_mask=None,
        negative_attention_mask=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and (prompt_embeds is None or generated_prompt_embeds is None):
            raise ValueError(
                "Provide either `prompt`, or `prompt_embeds` and `generated_prompt_embeds`. Cannot leave "
                "`prompt` undefined without specifying both `prompt_embeds` and `generated_prompt_embeds`."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_embeds is not None and negative_generated_prompt_embeds is None:
            raise ValueError(
                "Cannot forward `negative_prompt_embeds` without `negative_generated_prompt_embeds`. Ensure that"
                "both arguments are specified"
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if attention_mask is not None and attention_mask.shape != prompt_embeds.shape[:2]:
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {attention_mask.shape} != `prompt_embeds` {prompt_embeds.shape}"
                )

        if transcription is None:
            if self.text_encoder_2.config.model_type == "vits":
                raise ValueError("Cannot forward without transcription. Please make sure to" " have transcription")
        elif transcription is not None and (
            not isinstance(transcription, str) and not isinstance(transcription, list)
        ):
            raise ValueError(f"`transcription` has to be of type `str` or `list` but is {type(transcription)}")

        if generated_prompt_embeds is not None and negative_generated_prompt_embeds is not None:
            if generated_prompt_embeds.shape != negative_generated_prompt_embeds.shape:
                raise ValueError(
                    "`generated_prompt_embeds` and `negative_generated_prompt_embeds` must have the same shape when "
                    f"passed directly, but got: `generated_prompt_embeds` {generated_prompt_embeds.shape} != "
                    f"`negative_generated_prompt_embeds` {negative_generated_prompt_embeds.shape}."
                )
            if (
                negative_attention_mask is not None
                and negative_attention_mask.shape != negative_prompt_embeds.shape[:2]
            ):
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {negative_attention_mask.shape} != `prompt_embeds` {negative_prompt_embeds.shape}"
                )

    def construct_direction(self, embs_source: torch.Tensor, embs_target: torch.Tensor):
        """Constructs the edit direction to steer the image generation process semantically."""
        return (embs_target.mean(0) - embs_source.mean(0)).unsqueeze(0)

    def auto_corr_loss(self, hidden_states, generator=None):
        reg_loss = 0.0
        for i in range(hidden_states.shape[0]):
            for j in range(hidden_states.shape[1]):
                noise = hidden_states[i : i + 1, j : j + 1, :, :]
                while True:
                    roll_amount = torch.randint(noise.shape[2] // 2, (1,), generator=generator).item()
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2

                    # if noise.shape[2] <= 8:
                    #     break
                    break  # We first consider full latent
                    noise = F.avg_pool2d(noise, kernel_size=2)  # it is the problem
        return reg_loss

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(self.vocoder.config.model_in_dim) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
        pred_type = self.scheduler.config.prediction_type
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        if pred_type == "epsilon":
            return model_output
        elif pred_type == "sample":
            return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (0.5)
        elif pred_type == "v_prediction":
            return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

    @torch.no_grad()
    def forward_diffusion(
            self,
            generated_prompt_embeds=None,
            prompt_embeds = None,
            attention_mask =None,
            latents: Optional[torch.FloatTensor] = None,
            audio_path: str = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "np",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            lambda_auto_corr: float = 20.0,
            lambda_kl: float = 20.0,
            num_reg_steps: int = 5,
            num_auto_corr_rolls: int = 5,
            inject_attention_mask: bool = False,
            ):

        # device = self._execution_device
        device ="cuda:0"
        
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare embeddings
        if do_classifier_free_guidance == False:
            batch_size = latents.shape[0]
            generated_prompt_embeds = generated_prompt_embeds[batch_size:]
            prompt_embeds = prompt_embeds[batch_size:]
            attention_mask = attention_mask[batch_size:]

        # 5. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 6. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                # noise_pred = unet(        
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                    cross_attention_kwargs={"timestep": CURRENT_TIMESTEP},
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # regularization of the noise prediction
                with torch.enable_grad():
                    for _ in range(num_reg_steps):
                        if lambda_auto_corr > 0:
                            for _ in range(num_auto_corr_rolls):
                                var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                                # Derive epsilon from model output before regularizing to IID standard normal
                                var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                                l_ac = self.auto_corr_loss(var_epsilon, generator=generator)
                                l_ac.backward()

                                grad = var.grad.detach() / num_auto_corr_rolls
                                noise_pred = noise_pred - lambda_auto_corr * grad

                        if lambda_kl > 0:
                            var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                            # Derive epsilon from model output before regularizing to IID standard normal
                            var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                            l_kld = self.kl_divergence(var_epsilon)
                            l_kld.backward()

                            grad = var.grad.detach()
                            noise_pred = noise_pred - lambda_kl * grad

                        noise_pred = noise_pred.detach()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        inverted_latents = latents.detach().clone()

                
        return inverted_latents

    @torch.no_grad()
    def backward_diffusion(
            self,
            generated_prompt_embeds=None,
            prompt_embeds = None,
            attention_mask =None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 200,
            guidance_scale: float = 3.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            quality_boosting_steps = 0,
            # reverse_process: True = False,
            return_x_hat_0 = False,
            use_source_uncon=False,
            use_unet_uncon=False,
            inject_attention_mask = False,
            **kwargs):
        """
            generated_prompt_embeds: language embedding from GPT-2
            prompt_embeds: T5 embedding
            latents: x_T
        """
        

        # device = self._execution_device
        device ="cuda:0"
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # 3. Encode input prompt: prompt_embeds
        # 4. Prepare embeddings
        if do_classifier_free_guidance == False:
            batch_size = latents.shape[0]
            generated_prompt_embeds = generated_prompt_embeds[batch_size:]
            prompt_embeds = prompt_embeds[batch_size:]
            attention_mask = attention_mask[batch_size:]


        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables: latents
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order     

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance and (not use_source_uncon and not use_unet_uncon) else latents
                    )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Set global variable for time step
                CURRENT_TIMESTEP = t

                # predict the noise residual
                if (not use_source_uncon and not use_unet_uncon):
                    noise_pred = self.unet(    
                        latent_model_input,
                        t,
                        encoder_hidden_states=generated_prompt_embeds, 
                        encoder_hidden_states_1=prompt_embeds, 
                        encoder_attention_mask_1=attention_mask,
                        return_dict=False,
                        cross_attention_kwargs={"timestep": CURRENT_TIMESTEP},
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    if use_source_uncon:
                        noise_pred_uncond = self.unet(    
                                latent_model_input,
                                t,
                                encoder_hidden_states=generated_prompt_embeds[0].unsqueeze(0), 
                                encoder_hidden_states_1=prompt_embeds[0].unsqueeze(0), 
                                encoder_attention_mask_1=attention_mask[0].unsqueeze(0),
                                return_dict=False,
                                cross_attention_kwargs={'scale':0},
                            )[0]
                        noise_pred_text = self.unet(    
                                latent_model_input,
                                t,
                                encoder_hidden_states=generated_prompt_embeds[1].unsqueeze(0), 
                                encoder_hidden_states_1=prompt_embeds[1].unsqueeze(0), 
                                encoder_attention_mask_1=attention_mask[1].unsqueeze(0),
                                return_dict=False,
                                cross_attention_kwargs={'scale':1},
                            )[0]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)  
                    elif use_unet_uncon:
                        noise_pred_uncond = self.unet_uncond(    
                                latent_model_input,
                                t,
                                encoder_hidden_states=generated_prompt_embeds[0].unsqueeze(0), 
                                encoder_hidden_states_1=prompt_embeds[0].unsqueeze(0), 
                                encoder_attention_mask_1=attention_mask[0].unsqueeze(0),
                                return_dict=False,
                                cross_attention_kwargs={'scale':0},
                            )[0]
                        noise_pred_text = self.unet(    
                                latent_model_input,
                                t,
                                encoder_hidden_states=generated_prompt_embeds[1].unsqueeze(0), 
                                encoder_hidden_states_1=prompt_embeds[1].unsqueeze(0),
                                encoder_attention_mask_1=attention_mask[1].unsqueeze(0),
                                return_dict=False,
                                cross_attention_kwargs={'scale':1},
                            )[0]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if return_x_hat_0:
            return x_hat_0_lst, latents
        else:
            return latents

    @torch.no_grad()
    def quick_backward_diffusion(
            self,
            generated_prompt_embeds=None,
            prompt_embeds = None,
            attention_mask =None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 200,
            guidance_scale: float = 3.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            quality_boosting_steps = 0,
            # reverse_process: True = False,
            return_x_hat_0 = False,
            use_source_uncon=False,
            use_unet_uncon=False,
            inject_attention_mask = False,
            **kwargs):
        """
            generated_prompt_embeds: language embedding from GPT-2
            prompt_embeds: T5 embedding
            latents: x_T
        """


        device ="cuda:0"
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # 3. Encode input prompt: prompt_embeds
        # 4. Prepare embeddings
        if do_classifier_free_guidance == False:
            batch_size = latents.shape[0]
            generated_prompt_embeds = generated_prompt_embeds[batch_size:]
            prompt_embeds = prompt_embeds[batch_size:]
            attention_mask = attention_mask[batch_size:]

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables: latents
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order     
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance and (not use_source_uncon and not use_unet_uncon) else latents
                )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Set global variable for time step
            CURRENT_TIMESTEP = t

            # predict the noise residual
            if (not use_source_uncon and not use_unet_uncon):
                noise_pred = self.unet(    
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                    cross_attention_kwargs={"timestep": CURRENT_TIMESTEP},
                )[0]

            # perform guidance
            if do_classifier_free_guidance:
                if use_source_uncon:
                    noise_pred_uncond = self.unet(    
                            latent_model_input,
                            t,
                            encoder_hidden_states=generated_prompt_embeds[0].unsqueeze(0), 
                            encoder_hidden_states_1=prompt_embeds[0].unsqueeze(0), 
                            encoder_attention_mask_1=attention_mask[0].unsqueeze(0),
                            return_dict=False,
                            cross_attention_kwargs={'scale':0},
                        )[0]
                    noise_pred_text = self.unet(    
                            latent_model_input,
                            t,
                            encoder_hidden_states=generated_prompt_embeds[1].unsqueeze(0),
                            encoder_hidden_states_1=prompt_embeds[1].unsqueeze(0),
                            encoder_attention_mask_1=attention_mask[1].unsqueeze(0),
                            return_dict=False,
                            cross_attention_kwargs={'scale':1},
                        )[0]
                    # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif use_unet_uncon:
                    noise_pred_uncond = self.unet_uncond(    
                            latent_model_input,
                            t,
                            encoder_hidden_states=generated_prompt_embeds[0].unsqueeze(0), 
                            encoder_hidden_states_1=prompt_embeds[0].unsqueeze(0), 
                            encoder_attention_mask_1=attention_mask[0].unsqueeze(0),
                            return_dict=False,
                            cross_attention_kwargs={'scale':0},
                        )[0]
                    noise_pred_text = self.unet(    
                            latent_model_input,
                            t,
                            encoder_hidden_states=generated_prompt_embeds[1].unsqueeze(0), 
                            encoder_hidden_states_1=prompt_embeds[1].unsqueeze(0),
                            encoder_attention_mask_1=attention_mask[1].unsqueeze(0),
                            return_dict=False,
                            cross_attention_kwargs={'scale':1},
                        )[0]
                else:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)


        if return_x_hat_0:
            return x_hat_0_lst, latents
        else:
            return latents