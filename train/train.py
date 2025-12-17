from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
import numpy as np
import re
import regex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model import Encoder, Decoder, ReasoningPolicy

import argparse
import os

instruction = """
You are a helpful assistant that solves math problems step-by-step with clear reasoning.
Your task:
1. Break down the solution into clear, logical steps.
2. The last line of your response should be of the form "####Answer: $ANSWER" where $ANSWER is the answer to the problem.

**Question**:
{question}

**Answer**:
"""

def preprocess_gsm(example):
    answer = example["answer"].split("####")[-1].strip()
    example["solution"] = example["answer"].split("####")[0].strip()
    return example

# reparameterization trick: 주어진 평균과 로그 분산으로부터 latent z 샘플 생성
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)  # 표준편차 계산
    eps = torch.randn_like(std)    # 정규분포로부터 잡음 샘플링
    return mean + eps * std

# KL divergence를 계산하는 함수
def kl_divergence(mean, logvar):
    # 정규분포 N(0, I)를 prior로 가정했을 때, KL divergence 공식:
    # D_KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl)

def kl_divergence_two_gaussians(mu_policy, logvar_policy, mu_target, logvar_target):
    """
    KL(N(policy) || N(target)) = 0.5 * sum[ log(sigma_target^2/sigma_policy^2) + 
        (sigma_policy^2 + (mu_policy - mu_target)^2)/sigma_target^2 - 1 ]
    """
    # sigma^2 = exp(logvar)
    kld_element = 0.5 * (logvar_target - logvar_policy + 
                        (torch.exp(logvar_policy) + (mu_policy - mu_target).pow(2)) / torch.exp(logvar_target) - 1)
    # 배치별로 latent_dim에 대해 합산한 뒤 평균을 취함
    return torch.mean(torch.sum(kld_element, dim=1))

def train_cvae_step(encoder, decoder, optimizer_cvae, batch_q, batch_r):
    optimizer_cvae.zero_grad()
    
    # Encoder: (Q, R) -> latent 분포 파라미터 및 샘플 z_enc 생성
    _, enc_mean, enc_logvar = encoder(batch_q, batch_r)
    z_enc = reparameterize(enc_mean, enc_logvar)
    
    # Decoder: (Q, z_enc) -> R 임베딩 재구성
    r_pred = decoder(batch_q, z_enc)
    
    # Reconstruction loss (MSE)와 KL divergence 손실 계산
    recon_loss = F.mse_loss(r_pred, batch_r, reduction='mean')
    kl_loss = kl_divergence(enc_mean, enc_logvar)
    
    loss_cvae = recon_loss + args.kl_weight * kl_loss
    loss_cvae.backward()
    optimizer_cvae.step()
    
    return loss_cvae.item()

def train_policy_step(encoder, reasoning_policy, optimizer_policy, batch_q, batch_r):

    encoder.eval()
    
    if args.rationale_train_method == 'mse':

        with torch.no_grad():
            _, enc_mean, enc_logvar = encoder(batch_q, batch_r)
            z_enc = reparameterize(enc_mean, enc_logvar)
        encoder.train()  # 이후 다른 용도로 encoder가 필요한 경우 대비

        optimizer_policy.zero_grad()
        # Reasoning Policy: Q -> 예측 latent (policy_mean)
        _, policy_mean, _ = reasoning_policy(batch_q)
        
        # MSE 손실: encoder에서 생성한 z_enc (target)와 policy의 예측 값 비교
        rationale_loss = F.mse_loss(policy_mean, z_enc, reduction='mean')
        rationale_loss.backward()
        optimizer_policy.step()

    else:

        with torch.no_grad():
            _, target_mean, target_logvar = encoder(batch_q, batch_r)
        encoder.train()  # 이후 다른 용도로 encoder가 필요한 경우를 위해
        
        optimizer_policy.zero_grad()
        # Reasoning Policy: Q -> 예측 latent 분포 (policy_mean, policy_logvar)
        _, policy_mean, policy_logvar = reasoning_policy(batch_q)
        
        # KL divergence를 손실 함수로 사용하여 두 분포 간의 차이를 최소화
        rationale_loss = kl_divergence_two_gaussians(policy_mean, policy_logvar, target_mean, target_logvar)
        rationale_loss.backward()
        optimizer_policy.step()
        
    return rationale_loss.item()

def evaluate_test_set(encoder, decoder, reasoning_policy, test_loader, args):
    encoder.eval()
    decoder.eval()
    reasoning_policy.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_policy_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_q, batch_r in test_loader:
            batch_q = batch_q.to(args.device)
            batch_r = batch_r.to(args.device)
            # CVAE 평가: 인코더와 디코더를 통해 재구성 손실과 KL 손실 계산
            _, enc_mean, enc_logvar = encoder(batch_q, batch_r)
            z_enc = reparameterize(enc_mean, enc_logvar)
            r_pred = decoder(batch_q, z_enc)
            recon_loss = F.mse_loss(r_pred, batch_r, reduction='mean')
            kl_loss = kl_divergence(enc_mean, enc_logvar)
            # Reasoning Policy 평가: 학습 방식에 따라 손실 계산
            if args.rationale_train_method == 'mse':
                _, policy_mean, _ = reasoning_policy(batch_q)
                policy_loss = F.mse_loss(policy_mean, z_enc, reduction='mean')
            else:
                _, target_mean, target_logvar = encoder(batch_q, batch_r)
                _, policy_mean, policy_logvar = reasoning_policy(batch_q)
                policy_loss = kl_divergence_two_gaussians(policy_mean, policy_logvar, target_mean, target_logvar)
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    print("=== Test Set Evaluation ===")
    print(f"Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"KL Divergence: {avg_kl_loss:.4f}")
    print(f"Policy Loss: {avg_policy_loss:.4f}")

def main(args):

    embed_id = args.embed_model.split("/")[-1]

    if not args.load:

        # 1. Load and preprocess the data

        if args.data == "gsm":

            gsm_dataset = load_dataset("openai/gsm8k", "main", split="train")

            gsm_dataset = gsm_dataset.map(preprocess_gsm)
        
            question = [instruction.format(question=question) for question in gsm_dataset["question"]]
            rationale = gsm_dataset["solution"]
        
        elif args.data == "math":

            # Login using e.g. `huggingface-cli login` to access this dataset
            dataset1 = load_dataset("EleutherAI/hendrycks_math","algebra",split="train")
            dataset2 = load_dataset("EleutherAI/hendrycks_math","counting_and_probability",split="train")
            dataset3 = load_dataset("EleutherAI/hendrycks_math","geometry",split="train")
            dataset4 = load_dataset("EleutherAI/hendrycks_math","intermediate_algebra",split="train")
            dataset5 = load_dataset("EleutherAI/hendrycks_math","number_theory",split="train")
            dataset6 = load_dataset("EleutherAI/hendrycks_math","prealgebra",split="train")
            dataset7 = load_dataset("EleutherAI/hendrycks_math","precalculus",split="train")

            math_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, gsm_dataset]).shuffle()

            question = [instruction.format(question=question) for question in math_dataset["problem"]]
            rationale = math_dataset["solution"]
        
        elif args.data == "numina":

            dataset = load_dataset("AI-MO/NuminaMath-CoT")
            
            question = [instruction.format(question=question) for question in dataset["problem"]]
            rationale = dataset["solution"]

        print(">>>Data preprocessing complete!")

        # 2. Embed data

        if args.embed_model == "sentence-transformers/all-MiniLM-L6-v2":
            embed_model = SentenceTransformer(args.embed_model).to(args.device)

            print(">>> Embedding Model Loaded!")
            
            question_embeddings = embed_model.encode(question).to(args.device)
            rationale_embeddings = embed_model.encode(rationale).to(args.device)

        elif args.embed_model == "jinaai/jina-embeddings-v3":
            embed_model = SentenceTransformer(args.embed_model, trust_remote_code=True).to(args.device)

            print(">>> Embedding Model Loaded!")

            task = "separation"
            question_embeddings = embed_model.encode(question, task=task, prompt_name=task, truncate_dim=1024)
            rationale_embeddings = embed_model.encode(rationale, task=task, prompt_name=task, truncate_dim=1024)

        print("# Question embeddings shape:", question_embeddings.shape)
        print("# Rationale embeddings shape:", rationale_embeddings.shape)

        embed_output = args.output + "/embeddings"
        os.makedirs(embed_output, exist_ok=True)

        np.save(embed_output + f"/{args.data}_question_embeddings_{embed_id}.npy", question_embeddings)
        np.save(embed_output + f"/{args.data}_rationale_embeddings_{embed_id}.npy", rationale_embeddings)

        print(">>>Embeddings saved successfully!")

    # 3. CVAE & Rationale Model Training

    device = args.device

    embed_output = args.output + "/embeddings"
    question_embeddings = np.load(embed_output + f"/{args.data}_question_embeddings_{embed_id}.npy")
    rationale_embeddings = np.load(embed_output + f"/{args.data}_rationale_embeddings_{embed_id}.npy")
    
    print(">>>Embeddings loaded successfully!")

    q_dim = args.embed_dim        
    r_dim = args.embed_dim         
    latent_dim = args.latent_dim    

    encoder = Encoder(q_dim=q_dim, r_dim=r_dim, latent_dim=latent_dim).to(device)
    decoder = Decoder(q_dim=q_dim, latent_dim=latent_dim).to(device)
    reasoning_policy = ReasoningPolicy(q_dim=q_dim, latent_dim=latent_dim).to(device)

    optimizer_cvae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.cvae_lr)
    optimizer_policy = optim.Adam(reasoning_policy.parameters(), lr=args.rationale_lr)

    # cosine learning rate scheduler 생성
    num_epochs_cvae = args.cvae_epochs
    num_epochs_policy = args.rationale_epochs

    scheduler_cvae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cvae, T_max=num_epochs_cvae)
    scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, T_max=num_epochs_policy)


    # ----------------------------
    # 더미 데이터 및 DataLoader 구성
    # ----------------------------
    dataset = TensorDataset(torch.tensor(question_embeddings), torch.tensor(rationale_embeddings))
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----------------------------
    # Phase 1: CVAE (AutoEncoder) 학습
    # ----------------------------
    num_epochs_cvae = args.cvae_epochs
    print("=== Phase 1: CVAE Training ===")
    for epoch in range(num_epochs_cvae):
        total_loss = 0.0
        num_batches = 0
        for batch_q, batch_r in dataloader:

            batch_q = batch_q.to(device)
            batch_r = batch_r.to(device)

            loss = train_cvae_step(encoder, decoder, optimizer_cvae, batch_q, batch_r)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs_cvae}], CVAE Loss: {avg_loss:.4f}')
        scheduler_cvae.step()

    # ----------------------------
    # Phase 2: Reasoning Policy 학습 (AutoEncoder 훈련 후)
    # ----------------------------
    num_epochs_policy = args.rationale_epochs
    print("\n=== Phase 2: Reasoning Policy Training ===")
    for epoch in range(num_epochs_policy):
        total_loss = 0.0
        num_batches = 0
        for batch_q, batch_r in dataloader:
            
            batch_q = batch_q.to(device)
            batch_r = batch_r.to(device)

            loss = train_policy_step(encoder, reasoning_policy, optimizer_policy, batch_q, batch_r)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs_policy}], Policy Loss: {avg_loss:.4f}')
        scheduler_policy.step()

    print(">>>Training complete!")

    # Test 데이터셋에 대한 평가를 수행할 수 있습니다.

    # 5. Save the model

    checkpoint_path = args.output + "/checkpoint/"
    
    # checkpoint 저장: 모델 state_dict와 옵티마이저 state_dict 포함
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'reasoning_policy_state_dict': reasoning_policy.state_dict(),
        'optimizer_cvae_state_dict': optimizer_cvae.state_dict(),
        'optimizer_policy_state_dict': optimizer_policy.state_dict(),
        # 추가 정보 (예: 마지막 에포크, 손실값 등)도 포함 가능
        'epoch': num_epochs_cvae + num_epochs_policy
    }

    os.makedirs(checkpoint_path, exist_ok=True)

    if args.embed_model == "sentence-transformers/all-MiniLM-L6-v2":
        torch.save(checkpoint, checkpoint_path + f'{args.data}_allmini_{args.rationale_train_method}_{args.kl_weight}_z{args.latent_dim}_model_train.pth')
    else: # jinaai/jina-embeddings-v3
        torch.save(checkpoint, checkpoint_path + f'{args.data}_jina_{args.rationale_train_method}_{args.kl_weight}_z{args.latent_dim}_model_train.pth')

    print(">>>Model saved successfully!")

    # ----------------------------
    # Test set 평가 코드 추가
    # ----------------------------
    # 여기서는 전체 데이터셋의 10%를 Test set으로 분리하여 평가합니다.
    # from torch.utils.data import random_split
    # test_dataset = load_dataset("openai/gsm8k", "main", split="test[:10%]")

    # test_dataset = test_dataset.map(preprocess_gsm)

    # test_question = [instruction.format(question=question) for question in test_dataset["question"]]
    # test_rationale = test_dataset["solution"]

    # embed_model = SentenceTransformer(args.embed_model, trust_remote_code=True).to(args.device)

    # task = "separation"
    # question_embeddings = embed_model.encode(test_question, task=task, prompt_name=task, truncate_dim=1024)
    # rationale_embeddings = embed_model.encode(test_rationale, task=task, prompt_name=task, truncate_dim=1024)

    # dataset = TensorDataset(torch.tensor(question_embeddings), torch.tensor(rationale_embeddings))
    # batch_size = args.batch_size
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # evaluate_test_set(encoder, decoder, reasoning_policy, test_loader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CVAE model')
    parser.add_argument('--data', type=str, default="gsm" ,help='training data')
    parser.add_argument('--load', action="store_true", help='Load embeddings')
    parser.add_argument('--embed_model', type=str, default="jinaai/jina-embeddings-v3")
    parser.add_argument('--embed_dim', type=int, default=1024, help='Dimension of the embedding')
    parser.add_argument("--latent_dim", type=int, default=256, help="Dimension of the latent space")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--cvae_epochs', type=int, default=100, help='Number of epochs to train the CVAE')
    parser.add_argument('--cvae_lr', type=float, default=1e-4, help='Learning rate for the CVAE')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='KL divergence weight') 
    parser.add_argument('--rationale_train_method', type=str, default='kld', help='MSE or KLD for training the Rationale')
    parser.add_argument('--rationale_epochs', type=int, default=100, help='Number of epochs to train the Rationale')
    parser.add_argument('--rationale_lr', type=float, default=1e-4, help='Learning rate for the Rationale')
    parser.add_argument('--output', type=str, default="./output", help='Path to the output model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cpu or cuda)')
    args = parser.parse_args()

    main(args)



