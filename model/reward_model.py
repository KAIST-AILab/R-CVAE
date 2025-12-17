import argparse
import os
from datasets import load_dataset

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForTokenClassification, AutoTokenizer
from trl import PRMConfig, PRMTrainer

import torch

def main(args):
    # 1. Pretrained 모델과 토크나이저 로드

    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=2, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model,padding="max_length", truncation=True, max_length=256)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 데이터셋 로드 (Hugging Face datasets 라이브러리를 이용, split은 "train"으로 지정)
    dataset = load_dataset(args.dataset, split="train")
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)

    # 4. 학습 인자 설정
    training_args = PRMConfig(
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=1000,
        output_dir=None,
    )

    # 5. RewardTrainer 초기화
    trainer = PRMTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,  # tokenizer를 전처리 클래스로 사용
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # 6. 모델 훈련
    trainer.train()

    # 8. Hugging Face Hub에 모델과 토크나이저 업로드 (사전에 huggingface-cli login 필요)
    hub_id = f"{args.output_dir}/{args.model.split('/')[-1]}-RM"
    model.push_to_hub(hub_id)
    tokenizer.push_to_hub(hub_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct", help="Pretrained model name")
    parser.add_argument("--dataset", type=str, default="trl-lib/prm800k", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="jhn9803", help="Output directory for the model")
    args = parser.parse_args()
    main(args)
