# Unofficial Implementation of MAE (Masked Autoencoders Are Scalable Vision Learners) Using Huggingface Transformers
Simple and clean implementation of [Masked Autoencoders Are Scalable Vision Learners]() using Huggingface Transformers.

## Requirements
Our code works with the following environment.
- `torch==1.10.0`
- `torchvision==0.11.0`
- `transformers==4.12.5`
- `datasets==1.16.1`

## Datasets and Pre-trained Models
- Please download the dataset from [here](https://image-net.org/download.php) and then put the files in `data/imagenet`.
- [Optional] You can also download the models pre-trained by us from [here](https://drive.google.com/drive/folders/1KaslFLb3CRyUAllIyIhV9ndXvu6_7ykI?usp=sharing) with its training log shown [here](https://wandb.ai/zhjohnchan/MAE/reports/Pre-Training-Log--VmlldzoxNDAxNDAy?accessToken=dqh44wxkunumaa2ql95a2wtw0lkgfo9bdpq8garfu1arho4us4715wvazxe7xjp7).

## Pre-training
- `bash scripts/pretrain.sh` to pre-train the model on `ImageNet`.

## Fine-tuning (In tuning)
- `bash scripts/finetune.sh` to fine-tune the model on `ImageNet`.

## Acknowledgements
The code is based on Huggingface Transformers and some of the code is borrowed from [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch).
