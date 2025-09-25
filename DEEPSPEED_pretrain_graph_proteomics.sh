#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/irsyadadam/src_biomolecule_instruction_tuning"

# GPU Configuration
CUDA_GPU="0,1"

# Data Paths
DATA_PATH="/path/to/data/proteomics_finetune_conversations.json"
PATIENT_GRAPHS_DIR="/path/to/patient_graphs"
OUTPUT_DIR="/path/to/graph_llm/pretrain"


# Model Configuration
LLM_VERSION="lmsys/vicuna-7b-v1.5"
VISION_TOWER="graph_tower"
CONNECTOR_TYPE="mlp2x_gelu"
CONV_VERSION="llama"
MODEL_MAX_LENGTH=2048

# Graph Encoder Configuration
GRAPH_TOWER_TYPE="gat"
GRAPH_HIDDEN_SIZE=512
GRAPH_DROPOUT=0.3

# Training Hyperparameters
PER_DEVICE_BATCH_SIZE=100 
GRAD_ACCUM_STEPS=1      
LEARNING_RATE=2e-3
WEIGHT_DECAY=0.01 
WARMUP_RATIO=0.1 
SAVE_STEPS=1000

echo "Graph Encoder Proteomics Pretraining..."
echo "Model: $LLM_VERSION (LLM FROZEN for pretraining)"
echo "Vision Tower: $VISION_TOWER ($GRAPH_TOWER_TYPE)"
echo "Data: $DATA_PATH"
echo "Patient Graphs: $PATIENT_GRAPHS_DIR"
echo "Output: $OUTPUT_DIR"

deepspeed --include localhost:$CUDA_GPU --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path $DATA_PATH \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VISION_TOWER \
    --connector_type $CONNECTOR_TYPE \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --proteomics_mode True \
    --graph_tower_type $GRAPH_TOWER_TYPE \
    --graph_hidden_size $GRAPH_HIDDEN_SIZE \
    --graph_dropout $GRAPH_DROPOUT \
    --patient_graphs_dir $PATIENT_GRAPHS_DIR \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --fp16 True \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --lazy_preprocess True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --group_by_modality_length False \
    --training_recipe common \
    --tune_type_llm frozen \
    --tune_type_vision_tower full \
    --tune_type_connector full \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.95

echo "tensorboard logs: tensorboard --logdir $OUTPUT_DIR"
echo "Pretrained model saved to: $OUTPUT_DIR"