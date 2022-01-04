#!/bin/bash

PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

NUM_GPU=8
model_name_or_path=outputs/vit-base-patch16-224  # PATH to the downloaded pre-trained model
train_dir=data/imagenet/train/  # PATH to the imagenet train dir
validation_dir=data/imagenet/val  # PATH to the imagenet val dir
warmup_ratio=0.05
learning_rate=1e-3
weight_decay=0.05
adam_beta1=0.9
adam_beta2=0.999
lr_scheduler_type=cosine
label_smoothing_factor=0.1
num_train_epochs=100
per_device_train_batch_size=128
per_device_eval_batch_size=128
gradient_accumulation_steps=1
output_dir=outputs/vit-base-patch16-224-finetune

python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} --master_port ${PORT_ID} finetune.py \
    --dataset_name dataset_folder.py \
    --train_dir ${train_dir} \
    --validation_dir ${validation_dir} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${output_dir} \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --adam_beta1 ${adam_beta1} \
    --adam_beta2 ${adam_beta2} \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --label_smoothing_factor ${label_smoothing_factor} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --dataloader_num_workers 16 \
    --evaluation_strategy epoch \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy epoch \
    --save_total_limit 5