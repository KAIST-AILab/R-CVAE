#!/bin/bash

# 인자 설정
DATA="gsm" # "gsm", "math", "numina"

EMBED_MODEL="jinaai/jina-embeddings-v3" # jinaai/jina-embeddings-v3
EMBED_DIM=1024 # 384, 1024
EMBED_ID="jina"

BATCH_SIZE=256
CVAE_EPOCHS=100
CVAE_LR=1e-4
RATIONALE_EPOCHS=100
RATIONALE_LR=1e-4
OUTPUT="./output"

LATENT_DIM=256
KL_WEIGHT=1.0
RATIONALE_METHOD="kld"
DEVICE="cuda"

LOG_FILE="train_${EMBED_ID}_${DATA}_${RATIONALE_METHOD}_${KL_WEIGHT}_z${LATENT_DIM}.log"

# 변수 설정 로그 출력
{
  echo "=== Training CVAE Model ==="
  echo "DATA: $DATA"
  echo "EMBED_MODEL: $EMBED_MODEL"
  echo "EMBED_DIM: $EMBED_DIM"
  echo "LATENT_DIM: $LATENT_DIM"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "CVAE_EPOCHS: $CVAE_EPOCHS"
  echo "CVAE_LR: $CVAE_LR"
  echo "KL_WEIGHT: $KL_WEIGHT"
  echo "RATIONALE_METHOD: $RATIONALE_METHOD"
  echo "RATIONALE_EPOCHS: $RATIONALE_EPOCHS"
  echo "RATIONALE_LR: $RATIONALE_LR"
  echo "OUTPUT: $OUTPUT"
  echo "DEVICE: $DEVICE"
  echo "==========================="
} > "./log/cvae/${LOG_FILE}"

# 실행
nohup python train.py \
    --data "$DATA" \
    --load \
    --embed_model "$EMBED_MODEL" \
    --embed_dim $EMBED_DIM \
    --latent_dim $LATENT_DIM \
    --batch_size $BATCH_SIZE \
    --cvae_epochs $CVAE_EPOCHS \
    --cvae_lr $CVAE_LR \
    --kl_weight $KL_WEIGHT \
    --rationale_train_method "$RATIONALE_METHOD" \
    --rationale_epochs $RATIONALE_EPOCHS \
    --rationale_lr $RATIONALE_LR \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    > "./log/cvae/${LOG_FILE}" 2>&1 &
