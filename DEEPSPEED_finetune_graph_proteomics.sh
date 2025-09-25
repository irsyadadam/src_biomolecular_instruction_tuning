#!/bin/bash
# DEEPSPEED_finetune_graph_proteomics.sh

export PYTHONPATH="${PYTHONPATH}:/home/irsyadadam/src_biomolecule_instruction_tuning"

CUDA_GPU="0,1"

DATA_PATH="/path/to/data/proteomics_finetune_conversations.json"
PATIENT_GRAPHS_DIR="/path/to/patient_graphs"
PRETRAINED_MODEL_PATH="/path/to/graph_llm/pretrain"
OUTPUT_DIR="/path/to/graph_llm/finetune"

LLM_VERSION="lmsys/vicuna-7b-v1.5"
VISION_TOWER="graph_tower"
CONNECTOR_TYPE="mlp2x_gelu"
CONV_VERSION="llama"
MODEL_MAX_LENGTH=2048

GRAPH_TOWER_TYPE="gat"
GRAPH_HIDDEN_SIZE=512
GRAPH_DROPOUT=0.3

PER_DEVICE_BATCH_SIZE=100
GRAD_ACCUM_STEPS=1           
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.01            
WARMUP_RATIO=0.03            
SAVE_STEPS=500

LORA_R=64                    
LORA_ALPHA=128               
LORA_DROPOUT=0.1

echo "Stage 2: QLoRA Fine-tuning (Maximum Memory Efficiency)"
echo "Pretrained model: $PRETRAINED_MODEL_PATH"
echo "Vision Tower: $VISION_TOWER ($GRAPH_TOWER_TYPE)"
echo "Data: $DATA_PATH"
echo "Patient Graphs: $PATIENT_GRAPHS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Effective batch size: $((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * 2))"
echo "QLoRA: 8-bit quantization + LoRA adapters"

deepspeed --include localhost:$CUDA_GPU --master_port 29502 tinyllava/train/train.py \
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
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
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
    --training_recipe qlora_int8 \
    --tune_type_llm qlora \
    --tune_type_vision_tower frozen \
    --tune_type_connector full \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_bias none \
    --bits 8 \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.95

echo "Tensorboard logs: tensorboard --logdir $OUTPUT_DIR"
echo "QLoRA adapters saved to: $OUTPUT_DIR"