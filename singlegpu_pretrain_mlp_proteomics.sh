#!/bin/bash

export CUDA_VISIBLE_DEVICES=4


DATA_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/final_data/proteomics_pretrain_conversations.json"
PROTEOMICS_DATA_PATH="../biomolecule_instruction_tuning/data/filtered_proteomics/"
OUTPUT_DIR="/local/irsyadadam/biomolecular_instruction_tuning_data/mlp_llm/pretrain"

LLM_VERSION="lmsys/vicuna-7b-v1.5"
VISION_TOWER="mlp" 
CONNECTOR_TYPE="mlp2x_gelu"
CONV_VERSION="llama"
MODEL_MAX_LENGTH=2048

PER_DEVICE_BATCH_SIZE=128  
GRAD_ACCUM_STEPS=1           
LEARNING_RATE=1e-3
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
SAVE_STEPS=2500              


echo "Proteomics Pretraining..."
echo "Model: $LLM_VERSION (LLM FROZEN for pretraining)"
echo "Data: $DATA_PATH"
echo "Proteomics: $PROTEOMICS_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Effective batch size: $((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * 2))"
echo "Training only MLP tower + connector"

python -m tinyllava.train.train \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VISION_TOWER \
    --connector_type $CONNECTOR_TYPE \
    --data_path $DATA_PATH \
    --proteomics_data_path $PROTEOMICS_DATA_PATH \
    --conv_version $CONV_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --proteomics_mode True \
    --num_proteins 4792 \
    --mlp_tower_type mlp_3 \
    --mlp_hidden_size 256 \
    --mlp_dropout 0.3 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --bf16 True \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --group_by_modality_length False \
    --training_recipe common \
    --tune_type_llm frozen \
    --tune_type_vision_tower full \
    --tune_type_connector full \
    --attn_implementation flash_attention_2 \
    
echo "tensorboard logs: tensorboard --logdir $OUTPUT_DIR"
echo "Pretrained model saved to: $OUTPUT_DIR"
