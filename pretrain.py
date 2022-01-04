#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
import transformers
from PIL import Image
from dataclasses import dataclass, field
from datasets import load_dataset
from einops import rearrange
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    ViTFeatureExtractor,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from modeling import ViTConfig, MAEForMiM

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="nateraw/image-folder", metadata={"help": "Name of a dataset from the datasets package"}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    mask_ratio: Optional[float] = field(
        default=0.75,
        metadata={
            "help": "ratio of the visual tokens/patches need be masked."
        },
    )

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default="configs/mae-base-patch16-224-in21k.json",
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    decoder_config_name: Optional[str] = field(
        default="configs/mae-base-patch16-224-in21k_decoder.json",
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_path: str = field(default="configs/mae-base-patch16-224-in21k_preprocessor_config.json",
                                        metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the 'MAE Pre-Training' task.
    ds = load_dataset(data_args.dataset_name, data_dir=data_args.data_files)

    config = ViTConfig.from_json_file(model_args.config_name)
    decoder_config = ViTConfig.from_json_file(model_args.decoder_config_name)
    model = MAEForMiM(
        config=config,
        decoder_config=decoder_config
    )

    feature_extractor = ViTFeatureExtractor.from_json_file(model_args.feature_extractor_path)

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    class RandomMaskingGenerator:
        def __init__(self, input_size, mask_ratio):
            if not isinstance(input_size, tuple):
                input_size = (input_size,) * 2

            self.height, self.width = input_size

            self.num_patches = self.height * self.width
            self.num_mask = int(mask_ratio * self.num_patches)

        def __repr__(self):
            repr_str = "Maks: total patches {}, mask patches {}".format(self.num_patches, self.num_mask)
            return repr_str

        def __call__(self):
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            return mask
    _masked_position_generator = RandomMaskingGenerator(config.image_size // config.patch_size, data_args.mask_ratio)

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [_train_transforms(pil_loader(f)) for f in example_batch["image_file_path"]]
        example_batch["masks"] = [_masked_position_generator() for _ in example_batch["image_file_path"]]
        return example_batch

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(train_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        patch_masks = torch.tensor([example["masks"] for example in examples], dtype=torch.bool)

        # calculate the label to be predicted
        mean = feature_extractor.image_mean
        std = feature_extractor.image_std

        mean = torch.as_tensor(mean)[None, :, None, None]
        std = torch.as_tensor(std)[None, :, None, None]
        unnormed_pixel_values = pixel_values * std + mean

        squeezed_pixel_values = rearrange(unnormed_pixel_values, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                                          p1=config.patch_size, p2=config.patch_size)
        normed_pixel_values = (squeezed_pixel_values - squeezed_pixel_values.mean(dim=-2, keepdim=True)
                               ) / (squeezed_pixel_values.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        pixel_values_patch = rearrange(normed_pixel_values, 'b n p c -> b n (p c)')

        batch_size, _, channel = pixel_values_patch.shape
        labels = pixel_values_patch[patch_masks].reshape(batch_size, -1, channel)

        return {"pixel_values": pixel_values, "patch_masks": patch_masks, "labels": labels}

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        data_collator=collate_fn,
        tokenizer=feature_extractor
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
