#!/bin/bash

# Define variables
STUDY="p53"
MODEL="model"
SCRIPT="pipeline/03_run_linear_probe.py"
EMBEDDINGS_DIR_BASELINE="slide_embeds/slide_embeds_p53_only_class_new"
EMBEDDINGS_DIR_NEW="slide_embeds/slide_embeds_p53_class_head_new"

# Run the Python script
python "$SCRIPT" \
    --study "$STUDY" \
    --model $MODEL \
    --embeddings_dir_baseline "$EMBEDDINGS_DIR_BASELINE" \
    --embeddings_dir_new "$EMBEDDINGS_DIR_NEW" \
