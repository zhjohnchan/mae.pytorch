#!/bin/bash

PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

NUM_GPU=4
train_dir=data/imagenet/train/  # PATH to the imagenet train dir
config_name=configs/mae-base-patch16-224-in21k.json
decoder_config_name=configs/mae-base-patch16-224-in21k_decoder.json
feature_extractor_path=configs/mae-base-patch16-224-in21k_preprocessor_config.json
mask_ratio=0.75
warmup_ratio=0.025
learning_rate=1.5e-4
weight_decay=0.05
adam_beta1=0.9
adam_beta2=0.95
lr_scheduler_type=cosine
num_train_epochs=400
per_device_train_batch_size=256

output_dir=outputs/vit-base-patch16-224

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID pretrain.py \
    --dataset_name dataset_folder.py \
    --train_dir ${train_dir} \
    --config_name ${config_name} \
    --decoder_config_name ${decoder_config_name} \
    --feature_extractor_path ${feature_extractor_path} \
    --mask_ratio ${mask_ratio} \
    --output_dir ${output_dir} \
    --remove_unused_columns False \
    --do_train \
    --fp16 \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --adam_beta1 ${adam_beta1} \
    --adam_beta2 ${adam_beta2} \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --dataloader_num_workers 16 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy epoch \
    --save_total_limit 5