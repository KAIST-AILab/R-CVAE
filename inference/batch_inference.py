from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from model import Encoder, Decoder, ReasoningPolicy
from tqdm import tqdm
from math_verify import parse, verify

from vllm import LLM, SamplingParams

import argparse
import os
import json
import re
import random
import numpy as np

instruction = """
You are a helpful assistant that solves math problems step-by-step with clear reasoning.
Your task:
1. Break down the solution into clear, logical steps.
2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

**Question**:
{question}

**Answer**:
"""

def set_seed(seed: int):
    # 파이썬의 random 모듈 seed 설정
    random.seed(seed)
    
    # NumPy seed 설정
    np.random.seed(seed)
    
    # PyTorch seed 설정
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CUDNN 관련 설정: 재현성 확보를 위해 사용 (성능은 다소 저하될 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_verify_answer(pred, label):
    global split_error

    try:
        pred = pred.split("####Answer:")[-1]
    except:
        print("Split error")
        split_error += 1
        return False

    gold = parse(label)
    answer = parse(pred)

    return verify(gold, answer)

class TestDataset(Dataset):
    def __init__(self, questions, gold_answers):
        self.questions = questions
        self.gold_answers = gold_answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {"question": instruction.format(question=self.questions[idx]), "answer": self.gold_answers[idx]}

def main(args):

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test data loading

    if args.data == "openai/gsm8k":
        test_dataset = load_dataset(args.data, "main", split='test')
        questions = test_dataset["question"]
        gold_answers =  [answer.split("####")[-1].strip() for answer in test_dataset["answer"]]

    elif args.data == "EleutherAI/hendrycks_math":
        dataset1 = load_dataset("EleutherAI/hendrycks_math","algebra",split="test")
        dataset2 = load_dataset("EleutherAI/hendrycks_math","counting_and_probability",split="test")
        dataset3 = load_dataset("EleutherAI/hendrycks_math","geometry",split="test")
        dataset4 = load_dataset("EleutherAI/hendrycks_math","intermediate_algebra",split="test")
        dataset5 = load_dataset("EleutherAI/hendrycks_math","number_theory",split="test")
        dataset6 = load_dataset("EleutherAI/hendrycks_math","prealgebra",split="test")
        dataset7 = load_dataset("EleutherAI/hendrycks_math","precalculus",split="test")

        test_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7]).shuffle()

        questions = test_dataset["problem"]
        pattern = r'\\boxed\{([^}]*)\}'
        gold_answers = [re.search(pattern, solution).group(1) for solution in test_dataset["solution"]]

    elif args.data == "HuggingFaceH4/MATH-500":
        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        questions = test_dataset["question"]
        gold_answers = test_dataset["answer"]

    elif args.data == "Maxwell-Jia/AIME_2024":
        test_dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

        questions = test_dataset["Problem"]
        gold_answers = test_dataset["Answer"]

    test_dataset = TestDataset(questions, gold_answers)
    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n>>> Test data loading complete!\n")

    # 2. Define the models

    q_dim = args.embed_dim         # 질문 임베딩 차원
    r_dim = args.embed_dim         # 근거 임베딩 차원
    latent_dim = args.latent_dim    # 잠재 공간 차원 (reasoning skill)

    encoder = Encoder(q_dim=q_dim, r_dim=r_dim, latent_dim=latent_dim)
    decoder = Decoder(q_dim=q_dim, latent_dim=latent_dim)
    reasoning_policy = ReasoningPolicy(q_dim=q_dim, latent_dim=latent_dim)

    checkpoint = torch.load(args.cvae_ckpt)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    reasoning_policy.load_state_dict(checkpoint['reasoning_policy_state_dict'])

        # 기존 generator, generator_tokenizer는 vllm으로 대체됨
    model_name = args.model
    # vllm 모델 초기화 (device는 CUDA 또는 CPU)
    llm = LLM(model_name, 
              device=device, 
              max_model_len=4096, 
              dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization)

    embed_model = SentenceTransformer(args.embed_model, trust_remote_code=True)

    print("\n>>> Models loaded!\n")

    # 3. Inference

    # Batch Inference
    correct = 0
    split_error = 0
    total_samples = 0

    for batch in tqdm(test_loader):
        # 배치 내 질문과 정답 분리

        batch_questions = batch["question"]
        batch_gold_answers = batch["answer"]

        total_samples += len(batch_questions)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.num_samples 
        )

        outputs_batch = llm.generate(batch_questions, sampling_params=sampling_params)
        print("\n>>> Generating samples complete for batch!\n")

        # 질문에 대한 임베딩 (배치 처리)
        task = "separation"
        embed_questions = torch.Tensor(embed_model.encode(batch_questions, task=task, prompt_name=task, truncate_dim=args.embed_dim))
        
        # reasoning_policy가 배치 입력을 지원한다고 가정
        z_hats, rationale_means, _ = reasoning_policy(q=embed_questions)

        # 각 질문별 생성된 샘플 텍스트를 리스트로 정리
        answer_samples_list = []
        for output in outputs_batch:
            samples = [completion.text for completion in output.outputs]
            answer_samples_list.append(samples)

        # 모든 배치의 답변 샘플을 한 번에 임베딩하기 위해 평탄화
        flattened_samples = [sample for samples in answer_samples_list for sample in samples]
        embed_answer_samples = torch.Tensor(embed_model.encode(flattened_samples, task=task, prompt_name=task, truncate_dim=args.embed_dim))

        # 배치 내 각 질문에 대해 최적의 답변 선택
        for idx, (question, gold_answer, samples) in enumerate(
            zip(batch_questions, batch_gold_answers, answer_samples_list)
        ):
            # 현재 질문에 해당하는 샘플 인덱스 계산
            start_idx = idx * args.num_samples
            end_idx = (idx + 1) * args.num_samples
            answer_embeddings = embed_answer_samples[start_idx:end_idx]

            # 질문 임베딩을 각 샘플 개수만큼 복제
            q_embed = embed_questions[idx].unsqueeze(0).repeat(args.num_samples, 1)
            z, enc_mean, _ = encoder(q_embed, answer_embeddings)

            if args.method == "mse":
                # MSE 기준 최적 샘플 선택
                diff = ((z - z_hats[idx].unsqueeze(0)) ** 2).mean(dim=1)
                best_sample_idx = torch.argmin(diff).item()
                best_answer = samples[best_sample_idx]
            else:  # kld 방식
                diff = abs(enc_mean - rationale_means[idx]).mean(dim=1)
                best_sample_idx = torch.argmin(diff).item()
                best_answer = samples[best_sample_idx]

            result = extract_verify_answer(best_answer, gold_answer)
            correct += result

            # 메모리 해제 (필요시)
            del answer_embeddings, q_embed

        print(f"Accuracy: {correct / total_samples * 100:.2f}%", flush=True)

        # 배치 처리 후 임베딩 변수 해제
        del embed_questions, embed_answer_samples, answer_samples_list, flattened_samples, outputs_batch

    print(f"Split error: {split_error}", flush=True)
    print(f"Accuracy: {correct / len(test_dataset) * 100:.2f}%", flush=True)

    # path = args.output+args.data.split("/")[-1]
    # os.makedirs(path, exist_ok=True)
    # output_file = f"{path}/generated_answer.json"
    # with open(output_file, "w") as f:
    #     json.dump(generated_answers, f, indent=4)

    # print(f">>> Saved generated answers in <{output_file}>")
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with CVAE')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--data', type=str, default="openai/gsm8k" ,help='test data')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Test model')
    parser.add_argument('--cvae_ckpt', type=str, default="./output/checkpoint/kld_1.0_model_train.pth", help='Trained CVAE checkpoint')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max length for generation')
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument('--method', type=str, default='mse', help='Method for selecting the best sample')
    parser.add_argument('--output', type=str, default='./output/generated_answer/', help='Model generated answer')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for vllm')
    args = parser.parse_args()

    main(args)
