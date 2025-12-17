#!/bin/bash

SEED=42 # 42,43,44

DATA="openai/gsm8k" # "openai/gsm8k", "EleutherAI/hendrycks_math"
LOG_DATA="gsm"

MODEL="HuggingFaceTB/SmolLM2-1.7B-Instruct" # "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"
LOG_MODEL="smol" # "smol", "llama", "qwen", "mistral"

EMBED_MODEL="jinaai/jina-embeddings-v3" # "jinaai/jina-embeddings-v3" , "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE=2
NUM_SAMPLES=16

MAX_NEW_TOKENS=512

METHOD="mse"
OUTPUT="./output/generated_answer/"

# "./output/checkpoint/kld_1.0_model_train.pth", "./output/checkpoint/kld_4.0_model_train.pth", "./output/checkpoint/kld_7.0_z256_model_train.pth"
# "./output/checkpoint/jina_kld_1.0_z256_model_train.pth" 
CVAE_CKPT="./output/checkpoint/jina_kld_1.0_z256_model_train.pth" 
CVAE_TRAIN="kld_1.0"

LATENT_DIM=256 # 128, 256

TOP_P=0.9 # 0.9, 0.95
TEMPERATURE=0.7 # 0.6, 0.8

GPU_MEMORY=0.9 # 0.8, 0.9

LOG_FILE="cvae_z${LATENT_DIM}_${CVAE_TRAIN}_${LOG_MODEL}_${METHOD}_${LOG_DATA}_${MAX_NEW_TOKENS}tokens_seed${SEED}.log"

{
  echo "=== Inference Baselines: Run $RUN ==="
  echo "SEED: $SEED"
  echo "MODEL: $MODEL"
  echo "DATA: $DATA"
  echo "CVAE_CKPT: $CVAE_CKPT"
  echo "EMBED_MODEL: $EMBED_MODEL"
  echo "LATENT_DIM: $LATENT_DIM"
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

nohup python -u batch_inference.py \
    --seed $SEED \
    --data "$DATA" \
    --model "$MODEL" \
    --cvae_ckpt "$CVAE_CKPT" \
    --latent_dim $LATENT_DIM \
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
