#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/irsyadadam/src_biomolecule_instruction_tuning"

# CHECKPOINT_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/mlp_llm/finetune"
# OUTPUT_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/embeddings/mlp_embeddings.csv"

CHECKPOINT_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/node_llm/finetune"
OUTPUT_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/embeddings/node_llm.csv"

PROTEOMICS_DATA_PATH="/home/irsyadadam/biomolecule_instruction_tuning/data/filtered_proteomics/"
BATCH_SIZE=64

mkdir -p "$(dirname "$OUTPUT_PATH")"

CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --proteomics_data_path $PROTEOMICS_DATA_PATH \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE

echo "Extraction completed. Embeddings saved to: $OUTPUT_PATH"