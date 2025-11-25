# R-CVAE
Reasoning Test-time Compute with CVAE

# R-CVAE: Reasoning Test-time Compute with CVAE

**CVAE (Conditional Variational Autoencoder)**를 활용하여 추론 시점의 연산(Test-time Compute)을 모델링하고 최적화하기 위한 연구/구현

## 📂 프로젝트 구조 (File Structure)

```plaintext
.
├── train.py                # 메인 모델 학습 스크립트
├── train.sh                # 학습 실행 쉘 스크립트
├── model.py                # CVAE 모델 아키텍처 정의
├── reward_model.py         # 리워드 모델 (Reasoning 평가용) 정의
├── reward.sh               # 리워드 모델 실행/평가 스크립트
├── inference.py            # 단일 추론 스크립트
├── batch_inference.py      # 배치 단위 추론 스크립트
├── baseline_inference.py   # 베이스라인 모델 추론 스크립트
├── encoder_inference.py    # 인코더 추론 스크립트
├── visualize.py            # 결과 시각화 스크립트
├── visualization_analysis.ipynb # 시각화 및 분석용 노트북
├── dataset.ipynb           # 데이터셋 전처리 및 탐색용 노트북
├── run.sh                  # 전체 파이프라인 또는 메인 실행 스크립트
├── run_baseline.sh         # 베이스라인 실행 스크립트
├── run_batch.sh            # 배치 작업 실행 스크립트
├── run_encoder.sh          # 인코더 실행 스크립트
└── README.md
