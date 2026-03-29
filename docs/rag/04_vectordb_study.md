# 벡터 DB — FAISS & ChromaDB

> 참고: [FAISS 공식 문서](https://faiss.ai/), [ChromaDB 공식 문서](https://docs.trychroma.com/docs/overview/introduction)

---

## 1. 벡터 DB란?

임베딩 모델이 생성한 벡터를 저장하고, 유사도 기반으로 빠르게 검색할 수 있는 데이터베이스.

```
[일반 DB]  SELECT * WHERE name = '타이레놀'   → 정확한 키워드 매칭
[벡터 DB]  "두통에 좋은 약 알려줘"            → 의미적으로 가까운 문서 반환
```

RAG 파이프라인에서 벡터 DB는 **검색 엔진** 역할을 한다.

```
문서 → 임베딩 모델 → 벡터 → [벡터 DB 저장]
질문 → 임베딩 모델 → 벡터 → [벡터 DB 검색] → 유사 문서 반환
```

---

## 2. FAISS

Meta(Facebook AI Research)가 개발한 고성능 유사도 검색 라이브러리.
C++ 기반이며 Python 바인딩을 제공한다.

### 특징

- 수십억 개 벡터 처리 가능 (Meta 내부에서 **1.5조 개** 벡터 인덱싱)
- RAM을 초과하는 데이터셋 지원 (디스크 기반 인덱싱)
- GPU 가속 지원 (CUDA)
- **정확도 vs 속도** 트레이드오프 조절 가능

### 설치

```bash
conda install -c pytorch faiss-cpu   # CPU
conda install -c pytorch faiss-gpu   # GPU
```

---

### 인덱스 종류

FAISS는 데이터 규모와 목적에 따라 다른 인덱스를 선택해야 한다.

| 인덱스 | 적합한 규모 | 학습 필요 | 벡터 삭제 | 메모리 |
|--------|------------|-----------|-----------|--------|
| `IndexFlatL2` | 소규모 (~10만) | 불필요 | 가능 | 높음 |
| `IndexIVFFlat` | 중규모 (10만~1000만) | 필요 | 불가 | 중간 |
| `IndexIVFPQ` | 대규모 (1000만+) | 필요 | 불가 | 낮음 |
| `IndexHNSWFlat` | 범용 (속도/정확도 균형) | 불필요 | 불가 | 중간-높음 |

---

### 인덱스별 사용법

**① IndexFlatL2 — 완전 탐색, 소규모**

모든 벡터와 비교하는 브루트포스 방식. 가장 정확하지만 느림.

```python
import faiss
import numpy as np

d = 1024  # 벡터 차원 (bge-m3 기준)

index = faiss.IndexFlatL2(d)
index.add(vectors.astype('float32'))  # 벡터 추가

D, I = index.search(query.astype('float32'), k=5)
# D: 거리, I: 인덱스 번호
```

**② IndexIVFFlat — 클러스터 기반, 중규모**

벡터를 `nlist`개 클러스터로 나누고, 검색 시 가까운 클러스터만 탐색.
`nprobe`로 탐색 클러스터 수 조절 → 클수록 정확하지만 느림.

```python
import faiss

d = 1024
nlist = 100  # 클러스터 수

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

index.train(vectors.astype('float32'))  # 학습 필요
index.add(vectors.astype('float32'))

index.nprobe = 10  # 탐색할 클러스터 수
D, I = index.search(query.astype('float32'), k=5)
```

**③ IndexHNSWFlat — 그래프 기반, 범용 추천**

학습 불필요. 속도와 정확도 균형이 가장 좋음. 소~중규모 RAG에 적합.

```python
import faiss

d = 1024
M = 32  # 노드당 연결 수 (4~64, 클수록 정확하지만 메모리 증가)

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200  # 인덱스 빌드 품질
index.hnsw.efSearch = 50         # 검색 품질

index.add(vectors.astype('float32'))
D, I = index.search(query.astype('float32'), k=5)
```

---

### RAG에서 FAISS 활용 예시

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("BAAI/bge-m3")

# 의약품 문서 인덱싱
documents = [
    "타이레놀은 해열 및 진통 효과가 있는 아세트아미노펜 성분의 의약품입니다.",
    "이부프로펜은 소염진통제로 두통, 치통, 근육통에 효과적입니다.",
    "항생제는 의사의 처방 없이 복용하면 내성이 생길 수 있습니다.",
]

doc_vectors = model.encode(documents).astype('float32')

# FAISS 인덱스 생성 및 저장
d = doc_vectors.shape[1]  # 1024
index = faiss.IndexFlatL2(d)
index.add(doc_vectors)

# 검색
query = "두통이 심한데 어떤 약을 먹어야 하나요?"
query_vector = model.encode([query]).astype('float32')

D, I = index.search(query_vector, k=2)

print("검색 결과:")
for rank, idx in enumerate(I[0]):
    print(f"{rank+1}위: {documents[idx]}")
```

**출력:**
```
검색 결과:
1위: 이부프로펜은 소염진통제로 두통, 치통, 근육통에 효과적입니다.
2위: 타이레놀은 해열 및 진통 효과가 있는 아세트아미노펜 성분의 의약품입니다.
```

---

## 3. ChromaDB

오픈소스 벡터 DB. FAISS보다 설정이 간단하고 메타데이터 필터링이 내장되어 있어 RAG 프로토타이핑에 적합.

### 특징

- 내부적으로 **HNSW** 인덱스 사용
- 문서, 메타데이터, 벡터를 한 번에 저장
- 임베딩 모델 직접 연동 가능 (자동 임베딩)
- 메타데이터 기반 필터링 지원
- In-memory / 로컬 영구저장 / 클라우드 배포 모두 지원

### 설치

```bash
pip install chromadb
```

---

### 기본 사용법

```python
import chromadb

# 로컬 영구 저장 (데이터 유지)
client = chromadb.PersistentClient(path="./chroma_db")

# 컬렉션 생성
collection = client.create_collection("medicine_docs")

# 문서 추가 (임베딩은 자동 생성)
collection.add(
    documents=[
        "타이레놀은 해열 및 진통 효과가 있는 아세트아미노펜 성분의 의약품입니다.",
        "이부프로펜은 소염진통제로 두통, 치통, 근육통에 효과적입니다.",
        "항생제는 의사의 처방 없이 복용하면 내성이 생길 수 있습니다.",
    ],
    metadatas=[
        {"category": "진통제", "requires_prescription": False},
        {"category": "소염진통제", "requires_prescription": False},
        {"category": "항생제", "requires_prescription": True},
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 검색
results = collection.query(
    query_texts=["두통이 심한데 어떤 약을 먹어야 하나요?"],
    n_results=2
)

print(results['documents'])
```

### 메타데이터 필터링

```python
# 처방전 불필요한 약만 검색
results = collection.query(
    query_texts=["두통약 추천"],
    n_results=2,
    where={"requires_prescription": False}  # 메타데이터 필터
)
```

> FAISS는 메타데이터 필터링이 없어 별도 구현이 필요하지만,
> ChromaDB는 기본 내장되어 있어 의약품 서비스처럼 카테고리 필터가 필요한 경우 유용하다.

---

## 4. FAISS vs ChromaDB 비교

| 항목 | FAISS | ChromaDB |
|------|-------|----------|
| **개발사** | Meta | Chroma AI |
| **라이선스** | MIT | Apache 2.0 |
| **적합한 규모** | 대규모 (수억~수조) | 소~중규모 |
| **설정 난이도** | 높음 (인덱스 직접 선택) | 낮음 (자동 설정) |
| **메타데이터 필터링** | 없음 (직접 구현) | 기본 내장 |
| **영구 저장** | 직접 구현 (`faiss.write_index`) | `PersistentClient`로 간단 |
| **임베딩 자동 생성** | 없음 | 있음 (모델 연동 시) |
| **GPU 지원** | 있음 | 없음 |
| **RAG 프로토타이핑** | 불편 | 편리 |
| **프로덕션 대규모** | 적합 | 한계 있음 |

---

## 5. 우리 프로젝트 선택 기준

의약품 QnA RAG 시스템 기준:

| 고려 요소 | 내용 |
|----------|------|
| 데이터 규모 | 의약품 수천 ~ 수만 건 → 소 ~ 중규모 |
| 메타데이터 필터링 | 카테고리, 처방 여부 필터 필요 |
| 개발 편의성 | 프로토타입 단계 → 빠른 개발 중요 |
| **결론** | **ChromaDB** 가 현 단계에 적합 |

> 규모가 커지거나 성능 최적화가 필요해지면 FAISS 또는 Milvus로 이전 고려.

---

## 참고 자료

- [FAISS 공식 문서](https://faiss.ai/)
- [FAISS GitHub Wiki — 인덱스 선택 가이드](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [ChromaDB 공식 문서](https://docs.trychroma.com/docs/overview/introduction)
