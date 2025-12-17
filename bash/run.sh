#!/bin/bash

# 인자 설정
DATA="openai/gsm8k" # "openai/gsm8k", "EleutherAI/hendrycks_math"
MODEL="Qwen/Qwen2.5-14B-Instruct" # "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=32

NUM_SAMPLES=16
MAX_NEW_TOKENS=512

METHOD="mse"
OUTPUT="./output/generated_answer/"

CVAE_CKPT="./output/checkpoint/kld_4.0_z128_model_train.pth" # "./output/checkpoint/kld_4.0_model_train.pth", "./output/checkpoint/kld_4.0_z128_model_train.pth"
Z_DIM=128 # 128, 256

TOP_P=0.9 # 0.9, 0.95
TEMPERATURE=0.8 # 0.6, 0.8

GPU_MEMORY=0.9 # 0.8, 0.9

# 로그 관련 설정
CVAE_TRAIN="kld_4.0"
LOG_MODEL="qwen"
LOG_DATA="gsm"

# 3회 반복 실행
for RUN in {1..3}
do
  LOG_FILE="cvae_z${Z_DIM}_${CVAE_TRAIN}_${LOG_MODEL}_${METHOD}_${LOG_DATA}_${MAX_NEW_TOKENS}tokens_run${RUN}.log"

  {
    echo "=== Inference Baselines: Run $RUN ==="
    echo "MODEL: $MODEL"
    echo "DATA: $DATA"
    echo "CVAE_CKPT: $CVAE_CKPT"
    echo "EMBED_MODEL: $EMBED_MODEL"
    echo "Z_DIM: $Z_DIM"
    echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
    echo "TOP_P: $TOP_P"
    echo "TEMPERATURE: $TEMPERATURE"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "NUM_SAMPLES: $NUM_SAMPLES"
    echo "METHOD: $METHOD"
    echo "OUTPUT: $OUTPUT"
    echo "GPU_MEMORY: $GPU_MEMORY"
    echo "==============================="
  } > "./log/cvae/${LOG_MODEL}/${LOG_FILE}"

  echo ">>> Run ${RUN} 시작됨. 로그: ${LOG_FILE}"

  nohup python -u inference.py \
      --data "$DATA" \
      --model "$MODEL" \
      --cvae_ckpt "$CVAE_CKPT" \
      --z_dim $Z_DIM \
      --embed_model "$EMBED_MODEL" \
      --batch_size $BATCH_SIZE \
      --num_samples $NUM_SAMPLES \
      --max_new_tokens $MAX_NEW_TOKENS \
      --top_p $TOP_P \
      --temperature $TEMPERATURE \
      --method "$METHOD" \
      --output "$OUTPUT" \
      --gpu_memory_utilization $GPU_MEMORY \
      >> "./log/cvae/${LOG_MODEL}/${LOG_FILE}" 2>&1 &

  echo ">>> Run ${RUN} 종료됨."
done