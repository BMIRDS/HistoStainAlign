#!/bin/bash

# Define variables for the arguments
STUDY="p53"
CHECKPOINT_DIR_BASE="p53_tangle_checkpoints/tangle_p53_binary_4096_0.01_5e-05_5e-07_12_30_class_head_only_class_new"
CHECKPOINT_DIR_NEW="p53_tangle_checkpoints/tangle_p53_binary_4096_0.01_5e-05_5e-07_12_30_class_head_new"
MODEL="model_adjusted.pt"
DATASET_CSV="../prov-gigapath/dataset_csv/p53"
DATA_DIR="p53_tile_embeds_valid/"
INFO_CSV="data/final_p53_data.csv"

# Run the Python script with the specified arguments
python utils/check_paired_alignment.py \
  --study $STUDY \
  --checkpoint_dir_base $CHECKPOINT_DIR_BASE \
  --checkpoint_dir_new $CHECKPOINT_DIR_NEW \
  --model $MODEL \
  --dataset_csv $DATASET_CSV \
  --data_dir $DATA_DIR \
  --info_csv $INFO_CSV
