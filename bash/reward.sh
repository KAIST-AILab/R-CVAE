#!/bin/bash

# 기본 모델, 데이터셋, 출력 디렉토리 값을 설정합니다.
MODEL="mistralai/Mistral-Small-24B-Instruct-2501" # "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"
DATASET="trl-lib/prm800k"
BATCH_SIZE=2 # 2, 4, 8, 16 
GRADIENT_ACCUMULATION_STEPS=2
OUTPUT_DIR="jhn9803"

model_id="mistral" # "smol", "llama", "qwen", "mistral"

{
    echo "=== Inference Baselines ==="
    echo "MODEL: $MODEL"
    echo "DATASET: $DATASET"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "==============================="  
} > "./log/reward/${model_id}.log"

nohup python -u reward_model.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir "$OUTPUT_DIR" \
    >> "./log/reward/${model_id}.log" 2>&1 &

