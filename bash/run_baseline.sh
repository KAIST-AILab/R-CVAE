#!/bin/bash

# 인자 설정
SEED=42 # 42, 43, 44

DATA="EleutherAI/hendrycks_math" # "openai/gsm8k", "EleutherAI/hendrycks_math"
LOG_DATA="math"

#meta-llama/Llama-3.2-3B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct, "meta-llama/Llama-3.2-1B-Instruct"
MODEL="meta-llama/Llama-3.2-1B-Instruct" # "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"
LOG_MODEL="llama1b"

METHOD="top_k" # "greedy", "top_p", "best" , "top_k", "majority"

NUM_SAMPLES=16 # 4, 8, 16
BATCH_SIZE=64 # 32, 64

MAX_NEW_TOKENS=1024 # 512, 1024

TOP_P=0.9 # 0.9, 0.95
TEMPERATURE=0.7  # 0.6, 0.8

LOG_FILE="baseline_${LOG_MODEL}_${METHOD}_${LOG_DATA}_${MAX_NEW_TOKENS}tokens_seed${SEED}.log"

{
  echo "=== Inference Baselines: Run $RUN ==="
  echo "SEED: $SEED"
  echo "MODEL: $MODEL"
  echo "DATA: $DATA"
  echo "METHOD: $METHOD"
  echo "NUM_SAMPLES: $NUM_SAMPLES"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
  echo "TOP_P: $TOP_P"
  echo "TEMPERATURE: $TEMPERATURE"
  echo "==============================="
} > "./log/baseline/${LOG_MODEL}/${LOG_FILE}"

nohup python -u baseline_inference.py \
    --seed $SEED \
    --model "$MODEL" \
    --data "$DATA" \
    --method "$METHOD" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    >> "./log/baseline/${LOG_MODEL}/${LOG_FILE}" 2>&1 &

