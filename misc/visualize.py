import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from adjustText import adjust_text

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, concatenate_datasets

from model import Encoder

from sentence_transformers import SentenceTransformer

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

def main(args):

    embed_model = SentenceTransformer(args.embed_model)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    Q = dataset["question"]
    R = [answer.split("####")[0].strip() for answer in dataset["answer"]]

    question_embeddings = embed_model.encode(Q)
    rationale_embeddings = embed_model.encode(R)

    dataset = TensorDataset(torch.tensor(question_embeddings),torch.tensor(rationale_embeddings))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    q_dim = 384         # 질문 임베딩 차원
    r_dim = 384         # 근거 임베딩 차원
    latent_dim = 256    # 잠재 공간 차원 (reasoning skill)
    r_emb_dim = 384     # Decoder 출력 차원

    encoder = Encoder(q_dim, r_dim, latent_dim)
    checkpoint = torch.load(args.cvae_ckpt)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    encoder.eval()

    print(">>> Strat VAE Inference")

    # 데이터셋 전체를 VAE 인코더에 통과해 z를 얻음
    # -------------------------------------------
    all_z = []

    with torch.no_grad():
        for batch_q,batch_r in dataloader:
            z, _, _ = encoder(batch_q, batch_r)
            all_z.append(z.numpy())

    all_z = np.concatenate(all_z, axis=0)  # (num_samples, latent_dim)

    print(">>> Strat t-SNE visualization")

    # t-SNE를 이용해 2차원으로 축소
    # -----------------------------
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    z_2d = tsne.fit_transform(all_z)

    # KMeans 클러스터링으로 군집 분리
    n_clusters = 10  # 군집 개수, 적절히 조절 가능
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(z_2d)
    centroids = kmeans.cluster_centers_  # (n_clusters, 2)

    # 각 클러스터의 중심에서 가까운 3개 포인트만 인덱스를 표시
    closest_points_by_cluster = []
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_points = z_2d[cluster_indices]
        center = centroids[i].reshape(1, -1)
        distances = cdist(cluster_points, center)
        sorted_idx = np.argsort(distances.flatten())[:3]
        closest_indices = cluster_indices[sorted_idx]
        closest_points_by_cluster.append(closest_indices)

    # 시각화
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=150, marker='X', label='Centroids')

    # adjustText용 텍스트 객체 리스트
    texts = []

    # 텍스트 라벨 추가 (아직 adjustText는 호출 안 함)
    for cluster_indices in closest_points_by_cluster:
        for idx in cluster_indices:
            x, y = z_2d[idx]
            text = plt.text(x, y, str(idx), fontsize=9, color='black')
            texts.append(text)

    # 겹치지 않도록 자동 조정
    adjust_text(texts, only_move={'points':'y', 'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    plt.title("t-SNE + KMeans (Top-5 per Cluster, Adjusted Labels)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='best')
    plt.tight_layout()

    # 그림 저장
    directory_path = args.output
    os.makedirs(directory_path, exist_ok=True)
    output_path = directory_path+"tsne_visualization.png"
    plt.savefig(output_path, dpi=300)

    print(f">>> Saved t-SNE visualization in <{output_path}>")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CVAE model')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--cvae_ckpt', type=str, default="./output/checkpoint/kld_model_train.pth", help='Trained CVAE checkpoint')
    parser.add_argument('--output', type=str, default="./output/visualization/", help='Path to the output model')
    args = parser.parse_args()

    main(args)
