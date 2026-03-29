# 임베딩 모델 & 유사도 검색

> 참고: [HuggingFace BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [Sentence Transformers 공식 문서](https://sbert.net), [HuggingFace BERT 101](https://huggingface.co/blog/bert-101), [OpenAI Embeddings 공식 문서](https://platform.openai.com/docs/guides/embeddings)

---

## 1. 임베딩(Embedding)이란?

텍스트를 고차원 숫자 벡터로 변환하는 기술. 의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 배치된다.

```
"강아지를 키우는 방법" → [-0.067, 0.021, -0.037, ...] (1024차원)
"개를 훈련시키는 법"  → [-0.063, 0.019, -0.041, ...] (방향 유사)
"자동차 엔진 수리하기" → [0.912, -0.231, 0.445, ...] (방향 다름)
```

---

## 2. 임베딩 발전 과정

### 2-1. One-Hot Encoding

가장 단순한 텍스트 수치화 방법.

```
강아지 → [1, 0, 0, 0]
고양이 → [0, 1, 0, 0]
자동차 → [0, 0, 1, 0]
```

**한계:**
- 단어 수만큼 차원이 증가 (한국어 10만 단어 → 10만 차원)
- 모든 단어 쌍의 유사도가 0 → 의미 관계 표현 불가

---

### 2-2. Word2Vec

> *"비슷한 문맥에서 등장하는 단어는 비슷한 의미를 가진다"* (Distributional Hypothesis)

비슷한 문맥에서 등장하는 단어들의 벡터를 유사하게 학습.

```
"강아지가 밥을 먹었다"  →  강아지 ≈ 고양이 (같은 위치에 등장)
"고양이가 밥을 먹었다"
```

**결과:**
```
강아지 → [0.21, -0.54, 0.87, ...]  (100~300차원)
고양이 → [0.22, -0.51, 0.85, ...]  ← 유사!
자동차 → [0.91,  0.34, -0.22, ...] ← 다름
```

**한계:** 단어 하나에 벡터 하나 고정 → 동음이의어 처리 불가
```
"배가 고프다" → 배 = ?
"배를 탔다"   → 배 = ?   (Word2Vec은 둘 다 같은 벡터)
```

---

### 2-3. Attention 메커니즘

> *"단어의 의미는 주변 문맥에 따라 달라진다"*

각 단어가 문장 내 다른 단어들을 얼마나 참고할지 **가중치(Attention Score)** 를 계산.

```
"배를 탔다" → "배"가 각 단어에 부여하는 가중치:
  배  : 0.1
  를  : 0.2
  탔다: 0.7  ← 높은 가중치 → "선박"으로 해석
```

이 가중치를 반영해 문맥에 따라 다른 벡터를 생성 → 동음이의어 구분 가능.

---

### 2-4. Transformer

Attention 메커니즘을 기반으로 한 딥러닝 아키텍처. (Vaswani et al., 2017 "Attention is All You Need")

```
[Encoder]  입력 문장 이해
[Decoder]  출력 문장 생성
```

**Encoder 처리 흐름:**
```
입력 토큰
    ↓
Token Embedding + Positional Encoding
    ↓
Self-Attention      ← 문장 내 모든 단어 간 관계 계산
    ↓
Feed Forward Network
    ↓
(N번 반복)
    ↓
문맥이 반영된 벡터
```

**Self-Attention:** 외부 문장이 아닌 자기 문장 안에서 단어끼리 서로 참고.

---

### 2-5. BERT

Google AI (2018). Transformer의 **Encoder만** 사용한 사전학습 모델.

| 항목 | BERT-base | BERT-large |
|------|-----------|------------|
| Transformer 레이어 | 12 | 24 |
| Hidden Size | 768 | 1024 |
| Attention Heads | 12 | 16 |
| 파라미터 수 | 110M | 340M |

**사전학습 방법 2가지:**

**① MLM (Masked Language Model)**
```
입력: "강아지는 [MASK] 동물이다"
목표: [MASK] = "귀여운" 예측
```
전체 토큰의 15%를 무작위로 마스킹 → 양방향 문맥으로 예측 학습

**② NSP (Next Sentence Prediction)**
```
[연결되는 쌍]  "강아지는 귀여운 동물이다." + "많은 사람들이 강아지를 키운다."  → True
[무관한 쌍]   "강아지는 귀여운 동물이다." + "오늘 날씨가 맑다."             → False
```
문장 간 관계 이해 학습. MLM 50% + NSP 50% 동시 학습.

**학습 데이터:** Wikipedia(25억 단어) + BooksCorpus(8억 단어) = **총 33억 단어**

---

## 3. BGE-M3 모델

BERT를 기반으로 유사도 검색에 특화되게 추가 학습한 임베딩 모델. BAAI(Beijing Academy of AI) 개발.

### 모델 스펙

| 항목 | 내용 |
|------|------|
| Base Model | XLM-RoBERTa |
| 벡터 차원 | 1024 |
| 최대 입력 길이 | 8192 토큰 |
| 지원 언어 | 100개 이상 |
| 라이선스 | MIT |

### 3가지 검색 방식 지원

| 방식 | 설명 | 특징 |
|------|------|------|
| **Dense Retrieval** | 벡터 유사도 기반 검색 | 의미 기반, 일반적으로 사용 |
| **Sparse Retrieval** | 단어 가중치 기반 검색 | BM25와 유사, 키워드에 강함 |
| **ColBERT** | 토큰 단위 다중 벡터 검색 | 정밀도 높음 |

> RAG 파이프라인에서는 Dense + Sparse를 결합한 **Hybrid Search**가 권장됨.

### 사용 예시

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-m3")

sentences = [
    "강아지를 키우는 방법",
    "개를 훈련시키는 법",
    "자동차 엔진 수리하기",
]

vectors = model.encode(sentences)

# 코사인 유사도 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

base = vectors[0]
for i, s in enumerate(sentences):
    print(f"{cosine_similarity(base, vectors[i]):.4f} | {s}")
```

**출력:**
```
1.0000 | 강아지를 키우는 방법
0.7752 | 개를 훈련시키는 법     ← 단어는 다르지만 의미 유사
0.4673 | 자동차 엔진 수리하기   ← 관련 없음
```

---

## 4. 유사도 측정 방법

Sentence Transformers는 4가지 유사도 측정 방식을 지원한다.

| 방식 | 설명 | 범위 | 특징 |
|------|------|------|------|
| **Cosine Similarity** | 두 벡터 간 각도 | -1 ~ 1 | 가장 범용적, 기본값 |
| **Dot Product** | 벡터 내적 | 제한 없음 | 정규화된 벡터에서 Cosine과 동일, 더 빠름 |
| **Euclidean Distance** | 직선 거리 | 0 ~ ∞ | 음수로 반환 |
| **Manhattan Distance** | 좌표 차이의 합 | 0 ~ ∞ | 음수로 반환 |

**코사인 유사도 공식:**

```
cos(θ) = (A · B) / (|A| × |B|)

각도 0°   → cos(0°)   = 1.0   완전 동일
각도 90°  → cos(90°)  = 0.0   무관
각도 180° → cos(180°) = -1.0  반대 의미
```

> 정규화된 벡터에서는 Dot Product가 Cosine Similarity와 동일한 결과를 내면서 더 빠르게 동작한다.

---

## 5. RAG에서의 역할

```
[인덱싱 단계]
문서 → Embedding Model → 벡터 → Vector DB 저장

[검색 단계]
질문 → Embedding Model → 벡터 → Vector DB에서 유사도 검색 → 관련 문서 반환
```

임베딩 모델의 품질이 RAG 전체 성능에 직접적인 영향을 미침.
특히 한국어 RAG에서는 다국어를 지원하는 `bge-m3` 같은 모델 선택이 중요.

---

## 6. 임베딩 모델 비교 — OpenAI vs BGE-M3

### OpenAI text-embedding 모델

OpenAI에서 제공하는 API 기반 임베딩 모델.

| 모델 | 최대 차원 | 최대 토큰 | 가격 (1K 토큰) |
|------|-----------|-----------|----------------|
| `text-embedding-3-large` | 3072 | 8192 | $0.00013 |
| `text-embedding-3-small` | 1536 | 8192 | $0.00002 |
| `text-embedding-ada-002` (구버전) | 1536 | 8192 | $0.0001 |

`text-embedding-3` 계열은 `dimensions` 파라미터로 출력 차원을 줄일 수 있다.
예: `text-embedding-3-large`를 `dimensions=1024`로 사용해 벡터 크기 절약.

**사용 예시:**

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="두통이 심한데 어떤 약을 먹어야 하나요?"
)

embedding = response.data[0].embedding  # 1536차원 벡터
```

---

### OpenAI vs BGE-M3 비교

| 항목 | OpenAI text-embedding-3-large | BGE-M3 |
|------|-------------------------------|--------|
| **운영 방식** | API (유료) | 로컬 실행 (무료) |
| **벡터 차원** | 최대 3072 | 1024 |
| **최대 입력** | 8192 토큰 | 8192 토큰 |
| **한국어 지원** | 있음 | 있음 (100개 언어) |
| **비용** | 사용량 기반 과금 | 없음 |
| **속도** | API 레이턴시 존재 | 로컬 GPU/CPU 성능에 따라 다름 |
| **프라이버시** | 데이터 외부 전송 | 로컬 처리 |
| **MTEB 벤치마크** | 64.6% | 상위권 |

> **우리 프로젝트 선택:** BGE-M3
> - 의약품 데이터를 외부 API로 보내지 않아도 되는 점 (프라이버시)
> - 무료 로컬 실행
> - 한국어 성능 안정적

---

## 참고 자료

- [BAAI/bge-m3 모델 카드](https://huggingface.co/BAAI/bge-m3)
- [Sentence Transformers 공식 문서](https://sbert.net)
- [HuggingFace BERT 101](https://huggingface.co/blog/bert-101)
- [OpenAI Embeddings 공식 문서](https://platform.openai.com/docs/guides/embeddings)
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT 원논문 (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
