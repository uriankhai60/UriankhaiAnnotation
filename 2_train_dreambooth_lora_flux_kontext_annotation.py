# kontext train code 주석 버전
import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

# wandb 사용시 wandb를 임포트 하는 로직
if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothDataset(Dataset):
    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            class_prompt,
            class_data_root=None,
            class_num=None,
            repeats=1,
            center_crop=False,
            buckets=None,
            args=None,
            ):
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None # 이게 뭐지?
        self.class_prompt = class_prompt
        self.buckets = buckets

        # args로 dataset_name이 있으면 load_dataset 라이브러리를 임포트
        # 데이터셋 이름(dataset_name)이 지정된 경우
        if args.dataset_name is not None:
            from datasets import load_dataset
        
            # 데이터셋 로드
            # dataset_name = "imagefolder", dataset_config_name = "local_dir"
            # 위처럼 지정해서 학습도 가능함
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )

            # 데이터셋 하위의 train에서 컬럼이름들 가지고 온다
            column_names = dataset["train"].column_names

            # args로 입력된 컬럼 이름들과 비교하여 점검하는 로직
            if args.cond_image_column is not None and args.cond_image_column not in column_names:
                raise ValueError(f"args.cond_image_column={args.cond_image_column} not in {column_names}")
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"default image column is {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(f"args.image_column={args.image_column} not in {column_names}")
            # 인스턴 이미지를 로드 str 포멧으로 이미지가 정의되어 있어야 함.
            # 만약에 path형태로 입력하였다면 Image.open하는 코드가 있어야 함.
            instance_images = [dataset["train"][i][image_column] for i in range(len(dataset["train"]))]
            
            # I2I 트레이닝에서 컨디셔널 이미지를 로드하는 부분
            cond_images = None
            cond_image_column = args.cond_image_column
            if cond_image_column is not None:
                cond_images = [dataset["train"][i][cond_image_column] for i in range(len(dataset["train"]))]
                # 개수 점검
                assert len(instance_images) == len(cond_images)

            # 캡션 컬럼을 지정하지 않은경우
            if args.caption_column is None:
                logger.info("캡션 컬럼을 지정을 안함")
                self.custom_instance_prompts=None
            
            # 캡션 컬럼을 지정한 경우
            # 체크하고 문제 없으면 로드해서 custom_instance_prompts에 전부 할당
            else:
                if args.caption_column not in column_names:
                    raise ValueError(f"args.caption_column={args.caption_column}이 {column_names}에 없습니다")
                custom_instance_prompts = dataset["train"][args.caption_column]
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))

        # 데이터셋 이름이 지정되지 않은 경우 커스텀 인스턴스 프롬프트는 없는 것으로 통일함
        # 인스턴스 데이터 루트로부터 하위에 존재하는 이미지 파일들을 전부 읽어와서 이미지로 로드해서 데이터로 사용
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("instance image가 루트 디렉토리에 없음")

            instance_images = [path for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None
        
        # 멤버 변수에 instance_images와 cond_images를 할당
        self.instance_images = []
        self.cond_images = []
        for i, img in enumerate(instance_images):
            self.instance_images.extend(itertools.repeat(img, repeats))
            if args.dataset_name is not None and cond_images is not None:
                self.cond_images.extend(itertools.repeat(cond_images[i], repeats))
        
        # 이미지를 전처리해서 pixel_values, cond_pixel_values로 할당
        # 전처리는 args에서 선택한 버킷처리와 random_flip, center_crop등을 적용
        self.pixel_values = []
        self.cond_pixel_values = []
        for i, image in enumerate(self.instance_images):
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            dest_image=None
            if self.cond_images:
                dest_image = exif_transpose(self.cond_images[i])
                if not dest_image.mode == "RGB":
                    dest_image = dest_image.convert("RGB")

            width, height = image.size

            # 가장 가까운 bucket 탐색해서 인덱스를 가져옴
            bucket_idx = find_nearest_bucket(height, width, self.buckets)
            target_height, target_width = self.buckets[bucket_idx]
            self.size = (target_height, target_width)

            # 할당한 버킷을 사이즈를 기준으로 
            # `이미지`, `컨디션 이미지`에 transform을 진행함
            image, dest_image = self.paired_transform(
                image,
                dest_image=dest_image,
                size=self.size,
                center_crop=args.center_crop,
                random_flip=args.random_flip,
            )

            # 버켓 인덱스와 이미지를 pixel_values에 담음
            # (Image, idx)로 담음
            self.pixel_values.append((image, bucket_idx))
            # 컨디션 이미지가 있다면 그 항목은 cond_pixel_values에 담음
            if dest_image is not None:
                self.cond_pixel_values.append((dest_image, bucket_idx))
        
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        # 클래스 이미지가 있다면 이것도 멤버변수 `self.class_images_path`로 만들어 둠
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            # 데이터셋의 길이는 둘중의 큰 값으로 진행함
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        # 이미지 트랜스폼 정의
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        # `버킷인덱스`, `인스턴스이미지`, `컨디션이미지`, `인스턴스 프롬프트`를 기본으로 example에 담고
        # 클래스 이미지가 있는 경우 `클래스 이미지`와 `클래스 프롬프트`도 같이 내보내는 함수
        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["bucket_idx"] = bucket_idx
        
        # 컨디션 픽셀 값이 존재한다면 example에 인덱스 맞춰서 할당
        if self.cond_pixel_values:
            dest_image, _ = self.cond_pixel_values[index % self.num_instance_images]
            example["cond_images"] = dest_image
        
        # 커스텀 인스턴스 프롬프트가 존재하는 경우
        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt
        # 커스텀 인스턴스 프롬프트가 없는 경우
        else:
            example["instance_prompt"] = self.instance_prompt

        # 클래스 이미지가 존재하는 경우
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt
        
        return example
    
    def paired_transform(self, image, dest_image=None, size=(224, 224), center_crop=False, random_flip=False):
        # 리사이즈
        resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize(image)
        if dest_image is not None:
            dest_image = resize(dest_image)
        
        # 둘이 동일하게 센터 크롭
        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
            if dest_image is not None:
                dest_image = crop(dest_image)
        # 센터 크롭이 아닌 경우 랜덤 크롭
        else:
            i,j,h,w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i,j,h,w)
            if dest_image is not None:
                dest_image = TF.crop(dest_image, i, j, h, w)
        
        # 랜덤 플립이 들어간 경우
        if random_flip:
            do_flip = random.random() < 0.5
            if do_flip:
                image = TF.hflip(image)
                if dest_image is not None:
                    dest_image = TF.hflip(dest_image)
        
        # 텐서변환, 노말라이즈
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])
        image = normalize(to_tensor(image))
        if dest_image is not None:
            dest_image = normalize(to_tensor(dest_image))
        
        return (image, dest_image) if dest_image is not None else (image, None)
        

class BucketBatchSampler(BatchSampler):
    def __init__(self, 
                 dataset:DreamBoothDataset, 
                 batch_size:int, 
                 drop_last:bool=False
                 ):
        # 유효성 검사
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("fuck")
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 그룹 인덱스 셋팅
        # 인덱스를 집어넣어서 셋팅 좀 복잡해서 꼼꼼히 뜯어봐야만 이해 가능할것으로 보임
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        for idx, (_, bucket_idx) in enumerate(self.dataset.pixel_values):
            self.bucket_indices[bucket_idx].append(idx)
        self.sampler_len = 0
        self.batches = []

        # Pre-generate batches for each bucket
        for indices_in_bucket in self.bucket_indices:
            # Shuffle indices within the bucket
            random.shuffle(indices_in_bucket)
            # Create batches
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                self.batches.append(batch)
                self.sampler_len += 1  # Count the number of batches

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return self.sampler_len


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path:str, revision:str, subfolder:str,        
):
    '''path의 컨피그를 읽어서 모델의 class를 리턴하는 함수'''
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(class_one, class_two):
    '''텍스트 인코더 클래스를 받아서 체크포인트 로드한 이후 리턴하는 함수'''
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # `클래스 이미지`와 `인스턴스 이미지`들을 리스트에서 연결한 후 스택(배치 차원에서 확장)
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # 컨디션 이미지가 있다면 그것도 batch에 추가
    batch = {"pixel_values":pixel_values, "prompts":prompts}
    if any("cond_images" in example for example in examples):
        cond_pixel_values = [example["cond_images"] for example in examples]
        cond_pixel_values = torch.stack(cond_pixel_values)
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch.update({"cond_pixel_values": cond_pixel_values})
    return batch


def _encode_prompt_wth_clip(
        text_encoder,
        tokenizer,
        prompt:str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt:int = 1,    
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensor="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("토큰을 넣던가 아니면 토크나이즈를 넣던가")
    
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype

    # CLIPTextModel의 pooled output 결과를 사용함
    prompt_embeds = prompt_embeds.pooler_output # (b,n,d) -> (b, d)
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device) # (b, d)

    # 텍스트 임배딩을 num_images_per_prompt 만큼 복사
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1) # (b, r, d)
    prompt_embeds = prompt_embeds.view(batch_size*num_images_per_prompt, -1) # (b*r, d)
    return prompt_embeds # (b*r, d)


def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # 프롬프트 토크나이즈
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids=text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("fuck")
        
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    # 모듈의 dtype 확인
    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    
    # 프롬프트 임배딩
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # 프롬프트 임배딩의 쉐이프
    _, seq_len, _ = prompt_embeds.shape # (b, s, d)

    # 프롬프트 임배딩을 num_images만큼 복제
    # mps에서 호환되도록 로직을 구성함
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1) # (b*r, s, d)
    return prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt:str, max_sequence_length, device=None, num_images_per_prompt:int=1, text_input_ids_list=None):
    '''n개의 텍스트 인코더와 토크나이저를 받아서 토큰 임배딩을 리턴하는 함수'''
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    # (b, d) = (b, 768)
    pooled_prompt_embeds = _encode_prompt_wth_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    # (b, s, d) = (b, 512, 768)
    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    ):
        pass


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
    ):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)
    pipeline_args_cp = pipeline_args.copy()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    with torch.no_grad():
        prompt = pipeline_args_cp.pop("prompt")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt, prompt_2=None)
    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                **pipeline_args_cp,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="mode",
        choices=["sample", "mode"],
        help="VAE encoding mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="Validation image to use (during I2I fine-tuning) to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-kontext-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
        if args.cond_image_column is not None:
            raise ValueError("Prior preservation isn't supported with I2I training.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.cond_image_column is not None:
        assert args.image_column is not None
        assert args.caption_column is not None
        assert args.dataset_name is not None
        assert not args.train_text_encoder
        if args.validation_prompt is not None:
            assert args.validation_image is None and os.path.exists(args.validation_image)

    return args


def main(args):

    # wandb와 hub_token이 동시에 입력된 경우 차단을 하는 로직
    # 보안성때문이므로 주석처리
    # if args.report_to == "wandb" and args.hub_token is not None:
    #     raise ValueError(
    #         "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
    #         " Please use `hf auth login` to authenticate with the Hub."
    #     )

    # mps를 쓰면서 bf16을 선택한경우 차단을 하는 로직
    # if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
    #     # due to pytorch#99272, MPS does not yet support bfloat16.
    #     raise ValueError(
    #         "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
    #     )

    # 로깅을 할 디렉토리 패스 설정
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 분산학습과 관련된 초기 셋팅들
    # 엑셀러레이터에 프로젝트 디렉토리, 로깅 디렉토리를 설정
    # 어떤 스텝에서 사용되지 않은 파라미터가 있어도 학습이 멈추지 않도록 지시
    # mixed_precision, log_with(어디다 로깅할 것인가)등을 명시하는 로직.
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # mps 사용하지 않으므로 주석처리
    # mps(애플의 GPU)
    # if torch.backends.mps.is_available():
        # accelerator.native_amp = False

    # wandb 설치 검사하는 로직
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    # 디버깅을 위해서 logging에 베이스 컨피그 설정하는 부분
    # INFO 이상부터 타임스텝을 붙여서 보여줌.
    # 예시) 08/10/2025 21:07:12 - INFO - train_dreambooth - ***** Running training *****
    # CRITICAL (50): 시스템 중단급 치명적 오류
	# ERROR (40): 작업 실패를 일으키는 오류
	# WARNING (30): 주의 필요(잠재적 문제)
	# INFO (20): 일반 진행 상황/요약
	# DEBUG (10): 디버깅용 상세 정보
	# NOTSET (0): 레벨 미설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 프린트와 비슷한 동작이며 현재 상태를 main_process 뿐만 아니라 모든 프로세스에서 터미널에 프린트하는 로직
    logger.info(accelerator.state, main_process_only=False)
    
    # 메인/서브 프로세스의 verbosity를 다르게 셋팅
    # 메인이면 트랜스포머 워닝과 디퓨저는 인포부터 출력
    # 서브면 에러의 경우에만 출력
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 시드 설정
    if args.seed is not None:
        set_seed(args.seed)

    # 만약 선행지식 보존을 args에서 셋팅한경우, `클래스 이미지`를 만드는 로직
    # 파이프라인을 만들어서 셋팅한 개수만큼 `클래스 이미지`를 생성함
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        # 만약 클래스 이미지가 args에서 셋팅한 이미지의 개수보다 적다면 만드는 로직
        # args로 prior dtype을 셋팅한 데이터타입으로 인퍼런스 함
        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            transformer = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
            )
            pipeline = FluxKontextPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)
            # 새롭게 만든 이미지를 몇개 만들어야 하는지 계산
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"num of class images to sample: {num_new_images}")

            # 데이터셋/데이터로더 생성, 사용할 프롬프트는 args의 클래스 프롬프트임
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            # 데이터로더 래핑
            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            # 이미지 생성
            # 이미지에 대해서 hash(지문)을 만들어 파일명에 추가함.
            # 추후 중복제거 혹은 오버라이팅 방지를 위해서라고 생각됨.
            for example in tqdm(
                sample_dataloader, desc="Generation class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images
                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            free_memory()
    
    # 메인 프로세스에서 output_dir을 생성함
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        # 허브에 푸쉬하는것을 선택한경우 허브에 푸쉬함
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True
            ).repo_id

    # 토크나이저1,2 로드
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    # 텍스트 인코더 클래스 이름 가져옴
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    
    # 스케줄러 로드
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # 텍스트 인코더 로드
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

    # vae 로드
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant,)

    # 트랜스포머 블록 로드
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,)

    # 오직 로라 레이어만 트레이닝 하므로 다른것들의 grad를 off
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # mixed_precision training을 위해서 dtype 캐스팅
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # mac은 bf16을 지원 안함.
    # nvidia를 쓰므로 해당부분 주석처리
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 모델 캐스팅
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # 그라디언트 체크포인팅 셋팅
    # 기본적으로 transformer는 그라디언트 체크포인팅 셋팅.
    # args에서 train_text_encoder를 하였다면 text_encoder도 그라디언트 체크포인트 설정.
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # 로라로 학습할 타겟 모듈을 선정한 경우와 그렇지 않은경우.
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    # 특별한 설정이 없으면 기본으로 트레인되는 레이어들
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "proj_mlp",
        ]

    # 로라 웨이트들을 트랜스포머 레이어에 삽입함.
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    # 트랜스포머에 adapter 부착
    transformer.add_adapter(transformer_lora_config)
    # 만약 텍스트 인코더 트레인을 셋팅한 경우 해당 부분에 대해서 로라 레이어를 부착함.
    # 단, 텍스트 인코더 one에만 해당함.
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    # 엑셀레이터로 랩핑된 모델을 랩핑을 벗겨 리턴하는 함수
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # 훅을 만들고 붙이는 로직
    # 모델 훅
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError("fuck")
            
                weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
    
    # 로드 모델 훅
    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)
    
    # 훅 붙이기
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Ampere GPU에서 사용가능한 속도 올려주는 부분 하는게 좋음
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # LR 선형 스케일링
    # "배치사이즈", "어커뮬레이션_스텝", "프로세스_수"에 따라서 LR의 사이즈를 키움.
    # 왜냐하면, "배치사이즈", "어커뮬레이션_스텝", "프로세스_수"를 늘리면 update 횟수가 줄어듬.
    # 따라서 일반적으로 배치사이즈1에 맞춰서 선형으로 스케일링함.
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # fp16으로 mixed_precision이 셋팅된 경우 업데이트가 불안정한 경우가 있어
    # LoRA 레이어는 32로 upcast하는 부분
    # 그 외의 경우는 transformer와 동일한 dtype으로 트레인함
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        cast_training_params(models, dtype=torch.float32)
    
    # 트랜스포머 로라 파라미터 리스트로
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # 텍스트 인코더 파라미터 리스트로
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
    
    # 트랜스포머 lora 블록과 관련있는 파라미터만 가져옴
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}

    # 텍스트 인코더도 필요시하면 포함하고 아니면 트랜스포머 lora 파라미터만 넘김
    if args.train_text_encoder:
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
    else:
        params_to_optimize = [transformer_parameters_with_lr]
    
    # 옵티마이저 클래스 생성
    # adamw(original, 8bit), prodigy만 지원함 기본은 adamw
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"
    # adamw인 경우
    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )
    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    # prodigy인 경우
    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # 버킷 입력이 있다면 버킷들 정렬 업데이트
    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    # 데이터셋 생성
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        buckets=buckets,
        repeats=args.repeats,
        center_crop=args.center_crop,
        args=args,
    )

    # t2i, i2i 로거에 출력
    if args.cond_image_column is not None:
        logger.info("I2I fine-tuning enabled.")

    # 배치샘플러 = 어떤 샘플들이 한 미니배치로 묶일지”를 결정해, 배치 단위의 인덱스 목록을 순서대로 내놓는 클래스.
    # 같은 버킷의 이미지들을 미니배치로 묶어주는 데이터셋 클래스
    batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers = args.dataloader_num_workers,
    )

    # 트레인 인코더를 훈련 안하는 경우
    # 택스트 임배딩 구하는 함수를 미리 만들어놈
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids
    
    # 텍스트 인코더를 트레인 안하고 커스텀 인스턴스 프롬프트가 없는경우
    # 하나의 인스턴스 프롬프트를 택스트 임배딩으로 사용한다
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)

    # 만약 선행지식 보존을 선택하고 and 텍스트 인코더 트레인을 안한경우
    # 클래스 프롬프트도 임배딩을 미리 생성해 놓는다
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )
    
    # 앞서 임배딩을 만든경우 메모리 릴리즈를 위해서 삭제
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        text_encoder_one.cpu(), text_encoder_two.cpu()
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()
    
    # 커스텀 트레인 프롬프트가 아닌 경우(= 고정된 입력 프롬프트로 트레인하는 경우)
    if not train_dataset.custom_instance_prompts:
        # 텍스트 인코더 트레인 하는 경우가 아니라면
        # 앞에서 추출한 임배딩들과 text_ids를 사용함
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        
        # 텍스트 인코더 트레인 하는 경우
        # 프롬프트들을 토큰 변환까지만 함
        else:
            tokens_one = tokenize_prompt()
    
    # 커스텀 인스턴스 프롬프트를 사용하고, 텍스트 인코더 트레인을 하지 않는다면
    # 커스텀 인스턴스 프롬프트들을 미리 임배딩 변환하여 캐싱해둠
    elif train_dataset.custom_instance_prompts and not args.train_text_encoder:
        cached_text_embeddings = []
        for batch in tqdm(train_dataloader, desc="Embedding prompts"):
            batch_prompts = batch["prompts"]
            prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                batch_prompts, text_encoders, tokenizers
            )
            cached_text_embeddings.append((prompt_embeds, pooled_prompt_embeds, text_ids))
        
        if args.validation_prompt is None:
            text_encoder_one.cpu()
            text_encoder_two.cpu()
            del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
            free_memory()
    
    # vae.config에 기록된 factor들을 가져옴.
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # cond_image가 있는지 판단 태그
    has_image_input = args.cond_image_column is not None

    # cache latent를 셋팅한 경우
    # latents_cache와 cond_latents_cache에 픽셀인코딩 값을 캐싱해두는 로직
    if args.cache_latents:
        latents_cache = []
        cond_latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if has_image_input:
                    batch["cond_pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    cond_latents_cache.append(vae.encode(batch["cond_pixel_values"]).latent_dist)
        # validation_prompt가 없으면 vae를 삭제함
        # 왜냐하면 이후 과정들에서 텍스트 캐싱 레이턴트들만 필요함
        if args.validation_prompt is None:
            vae.cpu()
            del vae
            free_memory()

    # 웜업 스텝 스케일링(프로세스 숫자만큼 n배함)
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    
    # 맥스 스텝이 arg에 샛팅되지 않았다면
    # 프로세스당 할당되는 데이터의 개수를 구하고
    # 어커물레이션 스텝을 나눠서 프로세스당 실제 업데이트 스텝수를 구하고
    # 에폭 x 프로세스 수 x 프로세스당 실제 업데이트 스텝수
    if args.max_train_steps is None:
        len_train_dataloader_after_shading = math.ceil(
            len(train_dataloader) / accelerator.num_processes
            )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_shading / args.gradient_accumulation_steps
            )
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    # 맥스 스텝이 설정되었다면
    # 프로세스의 숫자만큼을 곱해서 `스케줄러를 위한 트레이닝 스텝수`에 할당
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    # 러닝레이트 스케줄러 가져옴
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler, # 모든 프로세스에서 업데이트 하는 회수
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 엑셀러레이터 프리페어
    # 텍스트 인코더 트레인 유무에 따라 랩핑 객체가 변경됨
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        (
            transformer,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    # 트레인로더에 있는 데이터 개수를 보고
    # 에폭당 업데이트 회수(스텝수)를 계산함
    # 에폭당 업데이트 수 = 데이터로더의 수 / 어커뮬레이션 스텝
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    # max_train_steps가 없다면
    # 이것들을 계산한다.
    # 스탭수/에폭 * 총 에폭
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning("...")
    
    # 트레인 에폭을 올림으로 다시 계산한다.
    # 트레이닝 루프에서 결국은 num_train_epochs을 기준으로 총 학습을 진행하고.
    # 각 스텝마다 max_train_steps를 보고 학습을 중단한다.
    # 총 에폭 = 맥스 스탭 / 업데이트수 / 에폭
    # 요약하면 순서는
    # step1) max_train_steps가 없다면 데이터 수로 계산.
    # step2) 1epoch당 몇개 스탭이 필요한지 num_update_steps_per_epoch 계산.
    # step2) 올림으로 총 트레인 에폭(num_train_epoch)을 계산.
    args.num_train_epochs = math.ceil(args.max_train_steps/num_update_steps_per_epoch)

    # 트랙커 이름 설정하고 config에 어규멘트들 올리기
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-kontext-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # 트레인 배치사이즈 구함
    # 총 배치 사이즈 = 트레인 배치사이즈 * 프로세스 수 * 그라디언트 어커뮬레이션 스탭
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # 학습 이어서 시작을 선택한 경우
    # 파일명을 룰베이스로 가지고 파싱하여 초기 글로벌 스텝을 지정한다.
    # output_dir 안에 checkpoint-<숫자> 폴더를 만들고 그 안에 파일을 넣어주는 방식으로 해야 동작함.
    if args.resume_from_checkpoint:
        # 파일명에 latest가 들어가는 경우와
        # 들어가지 않는 경우를 나눠서 동작한다.
        # latest라고 명시를 하지 않았다면 해당 path에서 불러오고
        # (ex) bbb/ccc.safetensors
        # path = bbb
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        # 명시를 했다면 checkpoint 폴더를 기준으로 룰로 파싱하여 마지막 패스를 가져온다.
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        # 패스가 없는경우 무시하고 새롭게 트레이닝한다.
        # 패스가 있다면 룰베이스로 스탭수를 추출, 글로벌 스텝으로 설정한다.
        # 주의해야할 것은 output_dir에서 path를 찾는다는 것이다.
        # 때문에 이어서 학습하려면
        # output_dir에 checkpoint-0에 safetensor로 셋팅해놔야 한다.
        if path is None:
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1]) # 글로벌 스텝을 가져온다.

            initial_global_step = global_step # 초기 글로벌 스텝으로 셋팅
            first_epoch = global_step // num_update_steps_per_epoch # 에폭을 계산

    # resume 없으면 initial_global_step은 0
    else:
        initial_global_step = 0

    # 프로그레시브 바 설정
    # 보여주는것은 max_train_steps
    # 초기 initial_global_step 셋팅하고
    # 메인 프로세스에서만 보여줌
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable= not accelerator.is_local_main_process, 
    )


    # 타임스탭을 기준으로 시그마를 구하는 함수
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # 트랜스 포머에 가이던스 있는지 확인을 한다.
    has_guidance = unwrap_model(transformer).config.guidance_embeds

    # 드디어 에폭을 도는 트레이닝
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
        
        # 트레인 로더에서 배치 단위로 학습
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # 각 이미지마다 프롬프트가 있는경우 인코딩해서 사용함
                # 만약 각 이미지마다 프롬프트가 없는 경우 repeat해서 사용함
                # 이 로직은 embeds를 만들기 위한 로직임
                if train_dataset.custom_instance_prompts:
                    # 트레인인코더 훈련을 안하는 경우
                    # 캐쉬된것이 있으니 캐싱 임배딩을 사용
                    if not args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = cached_text_embeddings[step]
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                        tokens_two = tokenize_prompt(tokenizer_two, prompts, max_sequence_length=args.max_sequence_length)
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts
                        )
                # 커스텀 인스턴스 프롬프트가 없는경우는 반복회수를 구하거나
                # 아니면 n번 반복해서 인코딩함
                else:
                    elems_to_repeat = len(prompts)
                    if args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[
                                tokens_one.repeat(elems_to_repeat, 1),
                                tokens_two.repeat(elems_to_repeat, 1),
                            ],
                            max_sequence_length=args.max_seqence,
                            device=accelerator.device,
                            prompt=args.instance_prompt,
                        )

                # 이미지 -> 레이턴트 변환 로직
                # 캐싱된것을 입력했으면 그거 사용하고
                # 레이턴트로 인코딩 함.
                if args.cache_latents:
                    if args.vae_encode_mode == "sample":
                        model_input = latents_cache[step].sample()
                        if has_image_input:
                            cond_model_input = cond_latents_cache[step].sample()
                    else:
                        model_input = latents_cache[step].mode()
                        if has_image_input:
                            cond_model_input = cond_latents_cache[step].mode()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    if has_image_input:
                        cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)
                    if args.vae_encode_mode == "sample":
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        if has_image_input:
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.sample()
                    else:
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        if has_image_input:
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
                
                # vae config를 이용하여 스케일링
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                if has_image_input:
                    cond_model_input = (cond_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                    cond_model_input = cond_model_input.to(dtype=weight_dtype)

                # vae 스케일링 팩터 계산
                # 정상적인 경우는`vae_scale_factor=8`로 고정이 됨
                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                # 레이턴트 이미지 아이디 계산
                # latent 사이즈를 2x2로 하니까 절반이 해상도로 넣어서 아이디 구하는 부분
                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2]//2,
                    model_input.shape[3]//2,
                    accelerator.device,
                    weight_dtype,
                )
                if has_image_input:
                    cond_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        cond_model_input.shape[0],
                        cond_model_input.shape[2]//2,
                        cond_model_input.shape[3]//2,
                        accelerator.device,
                        weight_dtype,
                    )

                    # 컨디션 아이디의 경우는 3번째 체널에 1을 붙임
                    cond_latents_ids[..., 0] = 1
                    # 이미지 아이디와 배치 차원에서 컨켓
                    latent_image_ids = torch.cat([latent_image_ids, cond_latents_ids], dim=0)
                
                # 노이즈 샘플링(model_input 과 동일한 사이즈의 랜덤 텐서)
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # model_input과 noise간의 시그마 블랜딩으로
                # 각 스텝에서의 noisy_model_input을 구함.
                # sigma는 직관적으로 노이즈 강도라고 생각하면 될것 같음.
                # sigma가 1이면 노이즈.
                # sigma가 0이면 GT.
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1], # 4 channel
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape
                if has_image_input:
                    packed_cond_input = FluxKontextPipeline._pack_latents(
                        cond_model_input,
                        batch_size=cond_model_input[0],
                        num_channels_latents=cond_model_input.shape[1],
                        height=cond_model_input.shape[2],
                        width=cond_model_input.shape[3],
                    )
                    packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_input], dim=1)

                # kontext는 항상 가이던스를 가지고 있음
                guidance=None
                if has_guidance:
                    guidance = torch.tensor(
                        [args.guidance_scale],
                        device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                
                # 백터 필드 예측
                # noise - gt의 백터를 예측값으로 가짐
                model_pred = transformer(
                    hidden_states = packed_noisy_model_input,
                    timestep=timesteps/1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                # 만약 컨디션이 있는경우는 해당부분을 자름
                if has_image_input:
                    model_pred = model_pred[:, :orig_inp_shape[1]]
                # 결과 언팩
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # 시그마에 따른 로스용 웨이팅을 구함
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas
                    )
                
                # 플로우 매칭 로스
                # 노이즈와 GT의 차이
                # 실제 인퍼런스 시에는 노이즈에 pred 값을 빼면서 진행할 것임
                # 예시) x3 = x4 - sigma * pred...
                target = noise - model_input 

                # 선행지식보존의 경우
                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()


                # 로스 계산
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float())**2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # 선행지식 보존의 경우 prior_loss를 더해서 계산함
                if args.with_prior_preservation:
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)

                # 다른 프로세스 싱크를 기다리고, 그라디언트 클리핑 적용함
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.step()
            
            # 싱크 맞을때까지 기다렸다
            # a. 전역변수 스텝(global_step)올리고
            # b. 체크포인트 저장
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step+=1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # 체크포인트 저장 전 리미트 확인하고 처리하는 코드
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) = args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                # 제거 코드
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path) # 스테이트 저장
                        logger.info(f"save state to {save_path}")

            logs = {"loss": loss.detach.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        # 밸리데이션 단계
        # 파이프라인 만들고 데이터인퍼런스해서 tracker에 업데이트
        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                if not args.train_text_encoder:
                    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)
                pipeline = FluxKontextPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {"prompt": args.validation_prompt}
                if has_image_input and args.validation_prompt:
                    pipeline_args.update({"image": load_image(args.validation_image)})
                
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )

                if not args.train_text_encoder:
                    del text_encoder_one, text_encoder_two
                    free_memory()

                images=None
                free_memory()

        # 학습을 모두 마친후 로라 레이어 저장하는 코드블록
        # 트랜스포머 블록 선택시 텍스트 인코더 블록을 언랩해서 저장
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            modules_to_save = {}
            transformer = unwrap_model(transformer)
            if args.upcast_before_saving:
                transformer.to(torch.float32)
            else:
                transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)
            modules_to_save["transformer"] = transformer
            
            # 텍스트 인코더를 트레인 하는 경우
            if args.train_text_encoder:
                text_encoder_one = unwrap_model(text_encoder_one)
                text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
                modules_to_save["text_encoder"] = text_encoder_one
            else:
                text_encoder_lora_layers = None
            
            # 로라를 저장하는 코드라인
            FluxKontextPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
                **_collate_lora_metadata(modules_to_save),
            )

            # 마지막 인퍼런스
            transformer = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_path,
                subfoler="transformer",
                revision=args.revision,
                variant=args.variant
            )
            pipeline = FluxKontextPipeline.from_pretrained(
                args.pretrained_model_name_path,
                transformer=transformer,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )

            # 로라 웨이트 로드
            pipeline.load_lora_weights(args.output_dir)

            # 라스트 인퍼런스
            image = []
            if args.validation_prompt and args.num_validation_images > 0:
                pipeline_args = {"prompt":args.validation_prompt}
                if has_image_input and args.validation_image:
                    pipeline_args.update({"image": load_image(args.validation_image)})
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    is_final_validation=True,
                    torch_dtype=weight_dtype,
                )
                del pipeline
                free_memory()

            if args.push_to_hub:
                save_model_card(
                    repo_id,
                    images=images,
                    base_model=args.pretrained_model_name_or_path,
                    train_text_encoder=args.train_text_encoder,
                    instance_prompt=args.instance_prompt,
                    validation_prompt=args.validation_prompt,
                    repo_folder=args.output_dir,
                )
                upload_folder(
                    repo_id = repo_id,
                    folder_path = args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )
            images = None
        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)