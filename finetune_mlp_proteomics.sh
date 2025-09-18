#!/bin/bash

# Stage 2: Fine-tuning with Proteomics Data (48GB GPU optimized)
# This stage fine-tunes the pretrained model for instruction following

export CUDA_VISIBLE_DEVICES=4

# Paths - UPDATE THESE
DATA_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/final_data/proteomics_finetune_conversations.json"
PROTEOMICS_DATA_PATH="../biomolecule_instruction_tuning/data/filtered_proteomics/"
PRETRAINED_MODEL_PATH="/local/irsyadadam/biomolecular_instruction_tuning_data/pretrain_output"  # Output from Stage 1
OUTPUT_DIR="/local/irsyadadam/biomolecular_instruction_tuning_data/finetune_output"

# Model settings - MUST match Stage 1 exactly
LLM_VERSION="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VISION_TOWER="mlp"  # This will be handled specially in the proteomics mode
CONNECTOR_TYPE="mlp2x_gelu"  # Better than linear for proteomics
CONV_VERSION="llama"  # Changed from 'pretrain' to 'llama' for instruction following
MODEL_MAX_LENGTH=2048

# Fine-tuning hyperparameters (different from pretraining)
PER_DEVICE_BATCH_SIZE=256     # Match stage 1 for consistency
GRAD_ACCUM_STEPS=1           
LEARNING_RATE=2e-5           # Much lower than pretraining (was 1e-3)
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
SAVE_STEPS=2500              

echo "🎯 Starting Stage 2: Proteomics Fine-tuning..."
echo "📚 Pretrained model: $PRETRAINED_MODEL_PATH"
echo "📊 Data: $DATA_PATH" 
echo "🔬 Proteomics CSVs: $PROTEOMICS_DATA_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "💾 Effective batch size: $((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))"

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
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
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
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --group_by_modality_length False \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_type_connector full \
    --attn_implementation flash_attention_2 \

echo "✅ Stage 2 fine-tuning completed!"
echo "📈 Check tensorboard logs: tensorboard --logdir $OUTPUT_DIR"
echo "📁 Fine-tuned model saved to: $OUTPUT_DIR"
echo "🚀 Your PPI-graph LLM is now ready for inference!"