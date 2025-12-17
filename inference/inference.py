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

instruction = """
You are a helpful assistant that solves math problems step-by-step with clear reasoning.
Your task:
1. Break down the solution into clear, logical steps.
2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

**Question**:
{question}

**Answer**:
"""

def extract_verify_answer(pred, label):
    global split_error

    try:
        pred = pred.split("####")[-1]
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
    # batch_size = args.batch_size
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n>>> Test data loading complete!\n")

    # 2. Define the models

    q_dim = 384         # 질문 임베딩 차원
    r_dim = 384         # 근거 임베딩 차원
    latent_dim = args.z_dim    # 잠재 공간 차원 (reasoning skill)
    r_emb_dim = 384     # Decoder 출력 차원

    encoder = Encoder(q_dim, r_dim, latent_dim)
    decoder = Decoder(q_dim, latent_dim, r_emb_dim)
    reasoning_policy = ReasoningPolicy(q_dim, latent_dim)

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

    embed_model = SentenceTransformer(args.embed_model)

    print("\n>>> Models loaded!\n")

    # 3. Inference

    correct = 0

    generated_answers = []
    split_error = 0

    for data in tqdm(test_dataset):
        question = data["question"]
        gold_answer = data["answer"]

        # vllm을 이용한 샘플 생성
        # print("\n>>> Generating samples with vllm...\n")
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.num_samples 
        )
        # vllm은 프롬프트 문자열 리스트를 입력받으며, 각 프롬프트에 대해 여러 완성을 생성할 수 있음
        outputs = llm.generate([question], sampling_params)

        print("\n>>> Generating samples complete!\n")
        
        answer_samples = []

        for i,completion in enumerate(outputs[0].outputs):
            answer_samples.append(completion.text)
        
        # 3. 임베딩 및 평가
        embed_question = torch.Tensor(embed_model.encode(question))
        rationale_input = torch.Tensor(embed_model.encode(question))
        z_hat, rationale_mean, rationale_logvar = reasoning_policy(rationale_input)

        embed_answer_samples = torch.Tensor(embed_model.encode(answer_samples))

        if embed_question.dim() == 1:
            embed_question = embed_question.unsqueeze(0)
        
        batch_q = embed_question.repeat(args.num_samples, 1)

        z, enc_mean, enc_logvar = encoder(batch_q, embed_answer_samples)

        if args.method == "mse":
            # 각 샘플별 평균 제곱 오차 계산 후 최소 오차 샘플 선택
            diff = ((z - z_hat.unsqueeze(0)) ** 2).mean(dim=1)
            best_sample_idx = torch.argmin(diff).item()
            best_answer = answer_samples[best_sample_idx]
        else: # kld
            # 각 샘플별 평균 차이를 계산 후 최소 차이 샘플 선택
            diff = abs(enc_mean - rationale_mean).mean(dim=1)
            best_sample_idx = torch.argmin(diff).item()
            best_answer = answer_samples[best_sample_idx]

        generated_answers.append({"question": question, "answer_samples": answer_samples, "best_answer": best_answer})

        result = extract_verify_answer(best_answer, gold_answer)
        correct += result

        # 메모리 해제
        del answer_samples, embed_answer_samples, embed_question, rationale_input

    print(f"Split error: {split_error}")
    print(f"Accuracy: {correct / len(test_dataset) * 100:.2f}%", flush=True)

    path = args.output+args.data.split("/")[-1]
    os.makedirs(path, exist_ok=True)
    output_file = f"{path}/generated_answer.json"
    with open(output_file, "w") as f:
        json.dump(generated_answers, f, indent=4)

    print(f">>> Saved generated answers in <{output_file}>")
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with CVAE')
    parser.add_argument('--data', type=str, default="openai/gsm8k" ,help='test data')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Test model')
    parser.add_argument('--cvae_ckpt', type=str, default="./output/checkpoint/kld_1.0_model_train.pth", help='Trained CVAE checkpoint')
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max length for generation')
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument('--method', type=str, default='mse', help='Method for selecting the best sample')
    parser.add_argument('--output', type=str, default='./output/generated_answer/', help='Model generated answer')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for vllm')
    args = parser.parse_args()

    main(args)

# Batch Inference
# correct = 0
#     total_examples = 0
#     num_samples = args.num_samples

#     for batch in tqdm(test_loader):
#         # 배치 내 질문과 정답 분리

#         batch_questions = batch["question"]
#         batch_gold_answers = batch["answer"]
#         current_batch_size = len(batch_questions)
#         total_examples += current_batch_size

#             # 질문 토크나이징 (패딩과 truncation 적용)
#         inputs = generator_tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True)
#         input_ids = inputs["input_ids"].to(generator.device)

#         # 각 질문의 실제 토큰 길이 계산 (패딩 제외)
#         # attention_mask는 1로 채워진 부분이 실제 토큰임
#         prompt_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch_size,)

#         # 각 질문에 대해 생성된 샘플마다 동일한 prompt_length가 적용되도록 반복
#         # flatten된 generator_outputs와 맞추기 위해 repeat_interleave 사용
#         prompt_lengths = prompt_lengths.repeat_interleave(num_samples)

#         print(">>>Generating samples...")

#         # 각 질문에 대해 num_samples 개의 답변 샘플 생성
#         generator_outputs = generator.generate(
#             input_ids,
#             num_return_sequences=num_samples,
#             do_sample=True,      # 샘플링 모드 활성화
#             max_new_tokens=args.max_new_tokens,
#         )

#         print(">>>Generating samples complete!")

#         # 전체 생성 답변 (총 current_batch_size * num_samples개)
#         # 각 출력 텐서에서 해당 질문의 실제 prompt_length 이후의 토큰만 디코딩
#         answer_samples_all = [
#             generator_tokenizer.decode(output[p_len.item():], skip_special_tokens=True)
#             for output, p_len in zip(generator_outputs, prompt_lengths)
#         ]

#         # 질문별로 num_samples개의 답변 샘플로 그룹화
#         answer_samples_batch = [
#             answer_samples_all[i * num_samples:(i + 1) * num_samples]
#             for i in range(current_batch_size)
#         ]

#             # 질문에 대한 임베딩 및 추론 정책 적용
#         embed_question = embed_model.encode(batch_questions)  # shape: (batch_size, embed_dim)
#         rationale_input = embed_model.encode(batch_questions)
#         z_hat, rationale_mean, _ = reasoning_policy(rationale_input)  # 각 질문별 결과

#         # 모든 생성 답변에 대한 임베딩 (총 current_batch_size*num_samples, embed_dim)
#         embed_answer_samples = embed_model.encode(answer_samples_all)

#         # 질문 임베딩을 각 샘플에 맞게 확장 (배치 차원 재구성)
#         batch_q_rep = embed_question.unsqueeze(1).repeat(1, num_samples, 1).view(current_batch_size * num_samples, -1)

#         # 인코더를 통해 질문-답변 쌍의 잠재 표현 계산
#         z, enc_mean, enc_logvar = encoder(batch_q_rep, embed_answer_samples)
#         z = z.view(current_batch_size, num_samples, -1)
#         enc_mean = enc_mean.view(current_batch_size, num_samples, -1)
#         enc_logvar = enc_logvar.view(current_batch_size, num_samples, -1)

#         # 선택 기준(method에 따라 다름)
#         if args.method == "mse":
#             # 각 질문별로 z와 z_hat 간의 평균 제곱 오차 계산
#             diff = ((z - z_hat.unsqueeze(1)) ** 2).mean(dim=2)  # shape: (batch_size, num_samples)
#             best_sample_idx = torch.argmin(diff, dim=1)             # 각 질문별 최적 샘플 인덱스
#         else:
#             # 예시: enc_mean와 rationale_mean의 차이를 절대값으로 계산하여 선택
#             diff = torch.abs(enc_mean - rationale_mean.unsqueeze(1)).mean(dim=2)
#             best_sample_idx = torch.argmin(diff, dim=1)

#         # 각 질문에 대해 선택한 샘플로 평가 진행
#         for i in range(current_batch_size):
#             best_answer = answer_samples_batch[i][best_sample_idx[i].item()]
#             gold_answer = batch_gold_answers[i]
#             result = extract_verify_answer(best_answer, gold_answer)

#             correct += result

#     print(f"Accuracy: {correct / total_examples * 100:.2f}%")