#!/bin/bash

# Define variables
CHECKPOINT_DIR="tangle_p53_binary_4096_0.01_5e-05_5e-07_12_30_class_head_only_class_new"
DATASET_CSV="../prov-gigapath/dataset_csv/p53/"
OUTPUT_DIR="slide_embeds_p53_only_class_new"
STUDY="p53"
MODEL_NAME="model.pt"
TILE_EMBED_DIR="../prov-gigapath/data/p53_embeddings/h5_files"
SCRIPT="pipeline/02_generate_slide_embeddings.py"

# Run the Python script
python "$SCRIPT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --dataset_csv "$DATASET_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --study "$STUDY" \
    --model_name "$MODEL_NAME" \
    --tile_embed_dir "$TILE_EMBED_DIR"
