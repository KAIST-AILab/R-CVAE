from datasets import load_dataset, concatenate_datasets
import math
from collections import Counter
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler

from math_verify import verify,parse

import os
import argparse
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

def majority_vote(preds, label):
    
    candidates = []
    
    for pred in preds:
        pred = pred.split("####Answer:")[-1]
        candidates.append(parse(pred)[0])

    try:
        gold = parse(label)[0]
    except:
        print("Gold parse error")
        return False
    
    votes = Counter(candidates)
    best_answer, _ = votes.most_common(1)[0]

    return verify(gold, best_answer)


def extract_verify_answer(pred, label):
    try:
        pred = pred.split("####Answer:")[-1]
    except:
        print("Split error")
        return False

    try:
        gold = parse(label)
    except:
        print("Gold parse error")
        return False

    try:
        answer = parse(pred)
    except:
        print("Answer parse error")
        return False
    
    print(f"Model Answer Parse: {answer}")
    print(f"Gold Answer Parse : {gold}")

    # Order here is important!
    return verify(gold, answer)

def main(args):

    set_seed(args.seed)

    if args.data == "openai/gsm8k":

        test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    elif args.data == "EleutherAI/hendrycks_math":

        dataset1 = load_dataset("EleutherAI/hendrycks_math","algebra",split="test")
        dataset2 = load_dataset("EleutherAI/hendrycks_math","counting_and_probability",split="test")
        dataset3 = load_dataset("EleutherAI/hendrycks_math","geometry",split="test")
        dataset4 = load_dataset("EleutherAI/hendrycks_math","intermediate_algebra",split="test")
        dataset5 = load_dataset("EleutherAI/hendrycks_math","number_theory",split="test")
        dataset6 = load_dataset("EleutherAI/hendrycks_math","prealgebra",split="test")
        dataset7 = load_dataset("EleutherAI/hendrycks_math","precalculus",split="test")

        test_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7]).shuffle()

    elif args.data == "HuggingFaceH4/MATH-500":

        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    elif args.data == "Maxwell-Jia/AIME_2024":

        test_dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model

    if args.method == "best":
        gpu_memory_utilization = 0.6
    else:
        gpu_memory_utilization = 0.9

    llm = LLM(model_name, 
        device=device, 
        max_model_len=4096, 
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization)

    print("\n>>> Models loaded!\n")

    if args.method == "best":

        # base_reward_model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
        # reward_tokenizer = AutoTokenizer.from_pretrained(args.model)
        # if reward_tokenizer.pad_token is None:
        #     reward_tokenizer.pad_token = reward_tokenizer.eos_token
        
        # if args.model == "HuggingFaceTB/SmolLM2-1.7B-Instruct":
        #     reward_model = PeftModel.from_pretrained(base_reward_model,"jhn9803/SmolLM2-1.7B-Instruct-RM")

        # elif args.model == "meta-llama/Llama-3.1-8B-Instruct":
        #     reward_model = PeftModel.from_pretrained(base_reward_model,"jhn9803/Llama-3.1-8B-Instruct-RM")

        # elif args.model == "Qwen/Qwen2.5-14B-Instruct":
        #     reward_model = PeftModel.from_pretrained(base_reward_model,"jhn9803/Qwen2.5-14B-Instruct-RM")

        # else:
        #     reward_model = PeftModel.from_pretrained(base_reward_model,"jhn9803/Mistral-Small-24B-Instruct-2501-RM")

        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
        rm = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                num_labels=1,
            )
        rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("\n>>> Reward Model loaded!\n")

        if args.data == "openai/gsm8k":

            questions = test_dataset["question"]
            gold_answers =  [answer.split("####")[-1].strip() for answer in test_dataset["answer"]]

        elif args.data == "EleutherAI/hendrycks_math":

            questions = test_dataset["problem"]
            pattern = r'\\boxed\{([^}]*)\}'
            gold_answers = [re.search(pattern, solution).group(1) for solution in test_dataset["solution"]]

        elif args.data == "HuggingFaceH4/MATH-500":

            questions = test_dataset["problem"]
            gold_answers = test_dataset["answer"]

        elif args.data == "Maxwell-Jia/AIME_2024":

            questions = test_dataset["Problem"]
            gold_answers = test_dataset["Answer"]

        correct = 0

        for question,gold_answer in tqdm(zip(questions, gold_answers)):
            
            sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.num_samples 
            )

            llm_outputs = llm.generate([question], sampling_params)

            print("\n>>> Generating samples complete!\n")

            answer_samples = []

            for i,completion in enumerate(llm_outputs[0].outputs):
                answer_samples.append(completion.text)

            # # 각 답변과 질문을 결합하여 입력 문자열 구성 (필요에 따라 형식을 조정)
            # input_texts = [question + "\n" + answer for answer in answer_samples]
            # inputs = reward_tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True).to(device)

            # with torch.no_grad():
            #     outputs = reward_model(**inputs)
            #     # 모델의 출력 형태에 따라 score 추출 (예: 로짓의 두 번째 클래스 혹은 단일 값)
            #     # 아래 예시는 로짓이 [batch_size, num_labels] 형태이고, 두 번째 클래스(logits[:, 1])가 긍정 점수라고 가정
            #     if outputs.logits.shape[-1] > 1:
            #         scores = outputs.logits[:, 1]
            #     else:
            #         scores = outputs.logits.squeeze(-1)
            # best_sample_idx = int(torch.argmax(scores, dim=0).item())
            # best_answer = answer_samples[best_sample_idx]

            input_texts = [rm_tokenizer.apply_chat_template([{"role": "user", "content": instruction.format(question=question)}, {"role": "assistant", "content": answer_sample}], tokenize=False) for answer_sample in answer_samples]
            inputs = rm_tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, ).to(device)
            
            with torch.no_grad():
                outputs = rm(**inputs)
                
                scores = outputs.logits.squeeze(-1)  # shape: [batch_size]
            
            best_index = torch.argmax(scores).item()
            best_answer = answer_samples[best_index]

            # best_answer = answer_samples[int(torch.argmax(torch.tensor(scores), dim=0).item())]

            result = extract_verify_answer(best_answer,gold_answer)
            correct += result

            del llm_outputs, outputs, inputs, scores, answer_samples
        
        print(f"Accuracy: {correct / len(questions) * 100:.2f}%")

    elif args.method == "majority":

        if args.data == "openai/gsm8k":

            questions = test_dataset["question"]
            gold_answers =  [answer.split("####")[-1].strip() for answer in test_dataset["answer"]]

        elif args.data == "EleutherAI/hendrycks_math":

            questions = test_dataset["problem"]
            pattern = r'\\boxed\{([^}]*)\}'
            gold_answers = [re.search(pattern, solution).group(1) for solution in test_dataset["solution"]]

        elif args.data == "HuggingFaceH4/MATH-500":

            questions = test_dataset["problem"]
            gold_answers = test_dataset["answer"]

        elif args.data == "Maxwell-Jia/AIME_2024":

            questions = test_dataset["Problem"]
            gold_answers = test_dataset["Answer"]

        correct = 0

        for question,gold_answer in tqdm(zip(questions, gold_answers)):

            sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.num_samples 
            )

            llm_outputs = llm.generate([question], sampling_params)

            print("\n>>> Generating samples complete!\n")

            answer_samples = []

            for i,completion in enumerate(llm_outputs[0].outputs):
                answer_samples.append(completion.text)

            result = majority_vote(answer_samples, gold_answer)
            correct += result

            del llm_outputs, answer_samples
        
        print(f"Accuracy: {correct / len(questions) * 100:.2f}%")

    else:

        all_generated_answers = []
        all_expected_answers = []

        num_batches = math.ceil(len(test_dataset) / batch_size)

        # 4. 배치 단위로 처리
        for i in tqdm(range(num_batches)):
            # 문제와 정답 배치 추출

            if args.data == "openai/gsm8k":

                questions = test_dataset["question"][i * batch_size : (i + 1) * batch_size]
                batch_expected = test_dataset["answer"][i * batch_size : (i + 1) * batch_size]

            elif args.data == "EleutherAI/hendrycks_math":

                questions = test_dataset["problem"][i * batch_size : (i + 1) * batch_size]
                batch_expected = test_dataset["solution"][i * batch_size : (i + 1) * batch_size]

            elif args.data == "HuggingFaceH4/MATH-500":

                questions = test_dataset["problem"][i * batch_size : (i + 1) * batch_size]
                batch_expected = test_dataset["answer"][i * batch_size : (i + 1) * batch_size]

            elif args.data == "Maxwell-Jia/AIME_2024":

                questions = test_dataset["Problem"][i * batch_size : (i + 1) * batch_size]
                batch_expected = test_dataset["Answer"][i * batch_size : (i + 1) * batch_size]

            if args.method == "greedy":

                # vllm 샘플링 파라미터 설정
                params = SamplingParams(
                            max_tokens=args.max_new_tokens,
                            temperature=1.0,
                        )
                
            elif args.method == "top_k":

                params = SamplingParams(max_tokens=args.max_new_tokens,
                                        temperature=args.temperature,
                                        top_k=32,
                                        )

            elif args.method == "top_p": # Top-p sampling
                
                params = SamplingParams(
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
            
            # vllm을 이용하여 배치 단위 문장 생성
            # 프롬프트 생성 (예: instruction 템플릿에 문제 삽입)
            if args.method == "beam":
                prompt_batch = [{"prompt": instruction.format(question=q)} for q in questions]
                outputs = llm.beam_search(prompt_batch, params)
                batch_generated = [output.sequences[0].text for output in outputs]
            else:    
                prompt_batch = [instruction.format(question=q) for q in questions]
                outputs = llm.generate(prompt_batch, params)
                batch_generated = [output.outputs[0].text for output in outputs]
            
            all_generated_answers.extend(batch_generated)
            all_expected_answers.extend(batch_expected)

            del outputs, prompt_batch, batch_generated, batch_expected

        if args.data == "openai/gsm8k":
            all_expected_answers = [item.split("####")[-1].strip() for item in all_expected_answers]

        elif args.data == "EleutherAI/hendrycks_math":
            pattern = r'\\boxed\{([^}]*)\}'
            all_expected_answers = [re.search(pattern, answer).group(1) for answer in all_expected_answers]

        correct = 0
        for idx,(gen,exp) in enumerate(zip(all_generated_answers, all_expected_answers)):
            print(">>> Verifying {idx} answer\n")

            result = extract_verify_answer(gen,exp)
            correct += result

        print(f"Accuracy: {correct / len(all_expected_answers) * 100:.2f}%", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--data", type=str, default="openai/gsm8k")
    parser.add_argument("--method", type=str, default="greedy")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    main(args)

