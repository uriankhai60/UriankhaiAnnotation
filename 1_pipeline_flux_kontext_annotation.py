# diffusers의 블랙포래스트랩 flux1.kontext pippeline.
# flux1.dev의 FluxImg2ImgPipeline과 FluxPipeline이 잘 믹스되어 있음.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# KONTEXT에서 선호하는 레졸루션.
# 유저가 입력된 width, height 중에서
# 가장 근사한 종횡비(aespect_ratio)를 가진 해상도로 인퍼런스 하도록 구현됨.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


# 이미지의 사이즈에 따라 mu을 조정하는 함수.
# 큰 이미지의 경우 mu 값이 커짐.
# 작은 이미지의 경우 mu 값이 약해짐.
def calculate_shift(
    image_seq_len,
    base_seq_len = 256,
    max_seq_len = 4096,
    base_shift = 0.5,
    max_shift = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# 스케줄러, mu(오프셋)와 sigmas(노이즈강도)를 받아서.
# 스케줄러를 통해 보정된 timesteps, num_inference_steps를 리턴함.
def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("timesteps나 sigmas 둘중 하나는 입력되어야 함.")
    
    if timesteps is not None:
        accept_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_timesteps:
            raise ValueError("스케줄러 안에 timesteps이 없음.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    # __call__ 정상분기에서는 sigmas, mu가 명시적으로 전달되어 옴.
    # 아래 분기가 표준 경로임.
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError("스케줄러 안에 sigmas가 없음.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps=scheduler.timesteps
        num_inference_steps=len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# VAE의 아웃풋인 encoder_output에 sample_mode의 형태로 latents값 수정하여 가져오는 함수.
def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    # sample이 표준경로임.
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("encoder_output의 attribute에 접근할 수 없음")

# 메인 파이프라인 클래스
class FluxKontextPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    # 오프로딩 적용시 오프로딩이 일어나는 순서
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    
    # 추가적인 컴포넌트들
    _optional_components = ["image_encoder", "feature_extractor"]
    # ㄴimage_encoder: 인코더 컴포넌트 (e.g.) ViT, ResNet과 같은 인코더
    # ㄴfeature_extractor: 전처리 컴포넌트 (e.g.) 모델 입력에 적절한 형태로 변환하는 전처리 컴포넌트
    
    # 콜백하는 텐서입력
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5TokenizerFast,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        # this.module = module을 대체하는 유틸함수
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        # 기본 vae_scale_factor는 8.
        # flux에서는 vae에서 4개 블록을 거치므로  vae_scale_factor=8(2**3)이 됨.
        # (e.g.) [1, 3, 1024, 1024] -> [1, 16, 128, 128]
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        # 기본 latent_channels는 16.
        # flux에서는 latent이후 channels가 고정된 값 16이 됨.
        # (e.g.) [1, 3, 1024, 1024] -> [1, 16, 128, 128]
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16

        # image_processor는 VaeImageProcessor(vae_scale_factor=16)이 됨.
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77)
        # 기본 샘플링 사이즈는 128.
        self.default_sample_size = 128


    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")


    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_images_per_prompts=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        # 디바이스와 타입 할당.
        # 디폴트는 _execution_device, text_encoder.dtype
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        # 스트링의 경우 리스트로 변환(디폴트 리스트 입력)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # 텍스트 인버전 기능이 활성화하는 경우.
        # 입력된 문자열해서 해당 토큰을 임배딩을 바꾸는 로직.
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        # 텍스트 -> 토큰(text_input_ids)
        text_input_ids=text_inputs.input_ids

        # 잘린 토큰이 있는 경우에 잘린부분 출력과 경고 출력하는 로직.
        # t5_encoder는 512 토큰임.
        untruncated_ids=self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text=self.tokenizer_2.batch_decode(untruncated_ids[:,self.tokenizer_max_length-1:-1])
            logger.warning(f"잘린 토큰(max_sequence_length:{max_sequence_length})는 {removed_text}")
        
        # 토큰->임배딩
        # [1, 512, 4096]
        prompt_embeds=self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype=self.text_encoder_2.dtype
        prompt_embeds=prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # 같은 프롬프트로 여러장의 이미지를 생성시, 임배딩을 복제
        prompt_embeds=prompt_embeds.repeat(1, num_images_per_prompts, 1)
        prompt_embeds=prompt_embeds.view(batch_size*num_images_per_prompts, seq_len, -1)

        # 임배딩 리턴
        return prompt_embeds
    

    def _get_clip_prompt_embeds(
        self,
        prompt,
        num_images_per_prompt=1,
        device=None,
    ):
        device=device or self._execution_device
        prompt=[prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt=self.maybe_convert_prompt(prompt, self.tokenizer)
        
        text_inputs=self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids=text_inputs.input_ids
        untruncated_ids=self.tokenzier(prompt, padding="longest", return_tensors="pt").input_ids 
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:,self.tokenizer_max_length-1:-1])
            logger.warning(f"CLIP의 max_length:{self.tokenizer_max_length}, removed_text:{removed_text}")
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # 전체 임배딩을 풀링해서 하나로 사용함.
        # 즉 문장에서 전체 컨텍스트를 대표하는 벡터를 만들어 사용하겠다는 의미로 해석됨.
        # [b, 768]
        prompt_embeds=prompt_embeds.pooler_output
        prompt_embeds=prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # [b, 768*n]
        prompt_embeds=prompt_embeds.repeat(1, num_images_per_prompt)
        # [b*n, 768]
        prompt_embeds=prompt_embeds.view(batch_size*num_images_per_prompt, -1)
        return prompt_embeds
    

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        # h,w,3의 제로 캔버스, 디폴트로는 128/2=64, 128/2=64가 들어옴
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]  # h를 따라서 1, 2, 3, 4, ...
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :] # w를 따라서 1, 2, 3, 4, ...
        # (64, 64, 3)
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape 
        # (4096, 3)
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels
        ) 
        return latent_image_ids.to(device=device, dtype=dtype)

    # pack을 하는 로직
    # [[1, 2],
    # [3, 4]]가 들어왔다면
    # [[[1, 2, 3, 4]]]로 2x2 patch에 맞춰서 플랫튼 함
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height//2, 2, width//2, 2) # (b, c, h//2, 2, w//2, 2)
        latents = latents.permute(0,2,4,1,3,5) # (0, h//2, w//2, c, 2, 2)
        latents = latents.reshape(batch_size, (height//2)*(width//2), num_channels_latents*4) # (0, h//2, w//2, c * 4)
        return latents

    # 레이턴트를 언팩하는 로직
    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        # b, n, c*4
        batch_size, num_patches, channels = latents.shape

        # latent의 height, width 계산 오리지널 width height의 //8
        height = 2 * (int(height) // (vae_scale_factor*2))
        width = 2 * (int(width) // (vae_scale_factor*2))

        # (b, h//2, w//2, c//4, 2, 2)
        latents = latents.view(batch_size, height//2, width//2, channels//4, 2, 2)
        
        # (b, c//4, h//2, 2, w//2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5) 
        
        # (b, c, h//2, w//2)
        latents = latents.reshape(batch_size, channels//(2*2), height, width)
        return latents

    # vae를 이용하여 이미지를 인코딩하는 함수
    def _encode_vae_image(self, image:torch.Tensor, generator:torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i:i+1]), generator=generator[i], sample_mode="argmax") for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        
        # 이미지 레이턴트 정구화 하는 부분
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        return image_latents


    def encode_prompt(
        self,
        prompt,
        prompt_2,
        device=None,
        num_images_per_prompt=1,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        max_sequence_length=512,
        lora_scale=None,
    ):
        device=device or self._execution_device

        # LoRA 레이어의 스케일 조정
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale=lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        # prompt를 list wrap
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        # 노말 루트, 프롬프트 임배드가 없는 경우
        if prompt_embeds is None:
            # prompt_2가 없다면 prompt를 prompt_2로 사용.
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # CLIPTextModel out의 pooled embed만 사용
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt, 
                num_images_per_prompt=num_images_per_prompt,
                device=device, 
                )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompts=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                )
        
        # LoRA 레이어의 스케일을 원래대로 복구
        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)
        
        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        
        return prompt_embeds, pooled_prompt_embeds, text_ids
    
    # diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_image 복사본
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
        
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds
    
    
    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator = None,
        latents = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("fuck")
        
        # 짝수이면서 8로 나누어떨어지는 노이즈 생성
        height = 2*(int(height) // (self.vae_scale_factor*2))
        width = 2*(int(width) // (self.vae_scale_factor*2))
        # shape = (B, 16, 128, 128)
        shape = (batch_size, num_channels_latents, height, width)

        # None 초기변수 선언
        image_latents = image_ids = None
        
        # I2I인 경우 들어온 이미지를 latent로 변환하는 로직
        if image is not None:
            # 이미지를 인코드
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            
            # 차원 추가
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError("fuck")
            else:
                image_latents = torch.cat([image_latents], dim=0)
            
            # I2I 이미지 레이턴트를 Flatten(_pack_latents)하는 로직
            # I2I 이미지 레이턴트의 포지셔널 인코딩 계산하는 로직
            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width)
            image_ids = self._prepare_latent_image_ids(batch_size, image_latent_height//2, image_latent_width//2, device, dtype)
            # 이미지 포지셔널 인코딩의 값을 통일
            image_ids[..., 0] = 1

        # 레이턴트의 포지셔널 인코딩을 구하는 부분(리턴을 위해)
        latent_ids = self._prepare_latent_image_ids(batch_size, height//2, width//2, device, dtype)

        # T2I에서 레이턴트 생성 및 계산
        if latents is None:
            # latents = (B, 16, 128, 128)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # latents 
            # (B, 16, 128, 128) -> (B, 64x64, 16*4) = (B, 4096, 64)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)
        
        return latents, image_latents, latent_ids, image_ids
    

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt
    ):
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image, device, 1)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if len(ip_adapter_image_embeds) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # 파이프라인 인퍼런스
    @torch.no_grad() # 그래디언트 중지
    def __call__(
        self,
        image = None,
        prompt = None,
        prompt_2 = None,
        negative_prompt = None,
        negative_prompt_2 = None,
        true_cfg_scale = 1.0,
        height = None,
        width = None,
        num_inference_steps = 28,
        sigmas = None,
        guidance_scale = 3.5,
        num_images_per_prompt = 1,
        generator = None,
        latents = None,
        prompt_embeds = None,
        pooled_prompt_embeds = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds = None,
        negative_ip_adapter_image = None,
        negative_ip_adpater_image_embeds = None,
        negative_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        joint_attention_kwargs = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ["latents"],
        max_sequence_length = 512,
        max_area = 1024**2,
        _auto_resize = True,
    ):
        # height, width 할당, 만약 없다면 128*8=1024사이즈로 할당함.
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # max_area 영역내에서 width, height를 다시 계산.
        original_height, original_width = height, width
        aspect_ratio = width/height
        width = round((max_area * aspect_ratio)**0.5)
        height = round((max_area / aspect_ratio)**0.5)

        # 16으로 나누어 떨어지는 해상도가 되도록 다시 width, height를 계산.
        multiple_of = self.vae_scale_factor * 2 # 16
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 만약 수정된 width, height가 original_width와 original_height와 다르다면 알람 호출
        if height != original_height or width != original_width:
            logger.warning(f"입력된 해상도 {original_width}x{original_height} -> {width}x{height}")

        # 적절한 입력이 들어왔는지 체크하는 함수
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 배치 개수를 결정하는 로직.
        # str이 들어오면 하나, [str, str, str, ...]이 들어오면 n개
        if prompt is not None and isinstance(prompt, str):
            batch_size=1
        elif prompt is not None and isinstance(prompt, list):
            batch_size=len(prompt)
        else:
            batch_size=prompt_embeds.shape[0]
        
        # .to("cuda") 혹은 from_pretrained(..., device_map=...)에서의 device_map을 따라감
        # 핼퍼 변수로써 해당 상태를 캐치해서 device로 가져옴.
        device = self._execution_device


        # LoRA 스케일을 set_adapters로 전달하는 게 기본
        # call을 할때 joint_attention_kwargs["scale"]는 그 위에 한 번 더 곱해서 조정가능함.
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # 네거티브 프롬이 있는지 체크
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )

        # 만약 네거티브 프롬이 있는경우
        # neg + s*(pos-neg)로 정식(=true)방법으로 가이딩을 주기 위함.
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 프롬프트 인코딩
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt = prompt,
            prompt_2 = prompt_2,
            prompt_embeds = prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        # 네거티브 프롬이 있으면 그것도 인코딩
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 이미지가 들어온 i2i의 경우 들어온 이미지를 전처리 하는 로직
        # flux에 맞도록 16의 배수로 사이즈를 맞추고(ratio를 유지한 상태로)
        # pt(pytorch tensor)로 변환하는 동작을 수행함
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            # kontext는 특정한 해상도에서만 훈련되었어서 그것에 맞는 해상도로 변환이 필요함
            # aspect_ratio의 차이가 가장 적은 해상도를 선택함
            if _auto_resize:
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w/h, w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width) # pt로 변환
        
        # 레이턴트 변수 준비
        # latent는 output_size = (1024, 1024)인 경우
        # (B, 16, 128, 128) -[패킹]-> (B, 64x64, 16x4) = (B, 4096, 64)가 됨
        # latent_ids의 경우 패킹된 하나의 영역마다 3개의 값을 가짐
        # (64, 64, 3), 영역당 가지고 있는 값은 [모달리티, y, x]
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # I2I인 경우 레이턴트 아이디를 이미지 아이디오 컨켓
        # 여기서부터 포지셔널인코딩의 값을 코드처럼 아아디라 지칭하겠음
        # concat((4096, 3), (4096, 3)) = (8192, 3)
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0) # dim=0

        # 타임스텝 준비
        sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # 4096 = (B, 4096, 64)
        image_seq_len = latents.shape[1]
        
        # 스케줄러에 맞춰 타입스탭과 인퍼런스 스텝을 조정
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 가이던스 조절
        # config에 guidance_embeds가 있다면 guidance를 만듬
        # 디폴트는 "guidance_embeds": true이므로 아래는 진행됨
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        # ip_adapter_image만 존재한다면 negative_ip_adapter 더미를 만들고
        # negatative_ip_adapter_image가 존재한다면 ip_adapter_image 더미를 만드는 로직.
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (negative_ip_adapter_image is None and negative_ip_adpater_image_embeds is None):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters
        
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (negative_ip_adapter_image is not None or negative_ip_adpater_image_embeds is not None):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters
        
        # joint_attention_kwargs가 없다면 더미 {}를 넣는 로직.
        # joint_attention_kwargs = 어텐션층에 전달되는 선택 인자 묶음.
        if self._joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}
        

        # ip_adapter 용도의 이미지와 네거티브 이미지의 임배딩을 계산하는 로직.
        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(ip_adapter_image, ip_adapter_image_embeds, device, batch_size*num_images_per_prompt)
        if negative_ip_adapter_image is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(negative_ip_adapter_image, negative_ip_adpater_image_embeds, device, batch_size*num_images_per_prompt)
        

        #  디노이징 루프
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            # 타임스탭만큼 루프를 회전
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # ip_adapter용도로 사용할 이미지 임배드가 있다면
                # _joint_attention_kwargs로 임배드를 밸류로 넣음
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                
                # 레이턴트 이름 업데이트
                latent_model_input = latents
                # 만약 kontext의 이미지 입력이 있다면 토큰을 컨켓.
                # (B, 4096, 64) -(cat)-> (B, 4096x2, 64)
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 노이즈 예측
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep/1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                # (핵심) 컨디션으로 concat된 노이즈는 자르고 randn으로 시작한 latents만 슬라이싱함
                noise_pred = noise_pred[:, :latents.size(1)]

                # 만약 do_true_cfg, 네거티브 프롬프트가 들어온 경우
                # 네거티브 노이즈를 구해서 그 값을 외삽한 값을 사용함
                # 네거티브 프롬프트가 존재할경우 노이즈 인퍼런스가 2회 이루어짐
                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep/1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states = negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:,:latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # 스케줄러를 통한 노이즈제거
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # mps에 관련된 내용이므로 삭제
                # if latents.dtype != latents_dtype:
                #     if torch.backends.mps.is_available():
                #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                #         latents = latents.to(latents_dtype)

                # 콜백함수가 주어진다면
                # 콜백의 입력 인자로 핵심 변수들(레이턴트, 프롬프트_임배드)를 전달하고
                # 콜백의 출력을 받아 교체하는 로직
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                
                if i == len(timesteps) - 1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order==0):
                    progress_bar.update()
        
        self._current_timestep = None
        
        # 사용자가 지정한 아웃풋 타입에 따라서
        # (a) latent 그대로, (b) 언팩후 이미지를 만들어서 리턴하는 로직
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        # 모델 오프로딩
        self.maybe_free_model_hooks()

        # 사용자가 튜플로 리턴을 원했으면 튜플로 리턴
        if not return_dict:
            return (image,)
        
        return FluxPipelineOutput(images=image)