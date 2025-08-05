#!/bin/bash

# Define variables
DATA_DIR="p53_tile_embeds_valid"
INFO_CSV="data/final_p53_data.csv"
STUDY="p53"
DATASET_CSV="../prov-gigapath/dataset_csv/p53"
N_TOKENS=4096

# Set CUDA devices and run the Python script
CUDA_VISIBLE_DEVICES=0,1,2,3 python pipeline/01_train_model_with_classification_head.py \
--data_dir "$DATA_DIR" \
--info_csv "$INFO_CSV" \
--study "$STUDY" \
--dataset_csv "$DATASET_CSV" \
--n_tokens "$N_TOKENS" \
--gpu_devices 0,1,2,3 \
--batch_size 12
