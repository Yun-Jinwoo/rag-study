# RAG 원논문 정리

> Lewis et al. (2020), *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*
> Facebook AI Research / UCL / NYU
> arXiv: [2005.11401](https://arxiv.org/abs/2005.11401)

---

## 1. 논문이 해결하려는 문제

LLM(대형 언어 모델)은 파라미터 안에 방대한 지식을 저장할 수 있지만, 두 가지 근본적 한계가 있다.

1. **지식 업데이트 불가** — 세상이 바뀌어도 모델을 재학습하지 않으면 최신 정보를 모름
2. **환각(Hallucination)** — 모르는 것을 그럴듯하게 지어냄
3. **출처 불투명** — 어디서 나온 답변인지 알 수 없음

이를 해결하기 위해 **파라미터 기억(Parametric Memory)** 과 **비파라미터 기억(Non-Parametric Memory)** 을 결합한 RAG 모델을 제안한다.

```
Parametric Memory     = BART (사전학습된 seq2seq 모델, 모델 가중치에 지식 저장)
Non-Parametric Memory = Wikipedia 벡터 인덱스 (외부 문서, 언제든 교체 가능)
```

---

## 2. 전체 아키텍처

```
질문 x
  ↓
[Retriever - DPR]  질문 벡터화 → FAISS로 Wikipedia에서 top-K 문서 검색
  ↓
  z₁, z₂, ..., zₖ  (검색된 문서 K개)
  ↓
[Generator - BART]  (x + zᵢ)를 입력으로 받아 답변 y 생성
  ↓
최종 답변 y
```

### 핵심 두 컴포넌트

| 컴포넌트 | 모델 | 역할 |
|----------|------|------|
| **Retriever** | DPR (Dense Passage Retriever) | 질문과 관련된 문서 검색 |
| **Generator** | BART-large (400M 파라미터) | 검색 문서 + 질문으로 답변 생성 |

---

## 3. 두 가지 RAG 모델 변형

### RAG-Sequence
- 검색된 문서 하나를 전체 출력 시퀀스에 동일하게 사용
- "이 문서 하나가 전체 답변을 책임진다"는 방식

```
p_RAG-Seq(y|x) ≈ Σ p(z|x) · p(y|x, z)
                  z∈top-K
```

### RAG-Token
- 출력 토큰마다 **다른 문서**를 참고할 수 있음
- 여러 문서에서 정보를 조합해 답변 생성 가능 → 더 유연함

```
p_RAG-Tok(y|x) ≈ Π Σ p(z|x) · p(yᵢ|x, z, y₁:ᵢ₋₁)
                  i  z∈top-K
```

> **언제 어떤 걸 쓰나?**
> - 단순 QA → RAG-Sequence가 유리한 경우 多
> - 여러 문서의 정보를 종합해야 할 때 → RAG-Token이 유리

---

## 4. Retriever: DPR (Dense Passage Retriever)

**Bi-Encoder 구조** — 질문과 문서를 각각 독립적으로 인코딩

```
d(z) = BERT_document(z)   ← 문서 벡터
q(x) = BERT_query(x)      ← 질문 벡터

유사도 = d(z)ᵀ · q(x)  (내적)
```

- 검색 속도를 위해 **FAISS** (Facebook AI Similarity Search) 사용
  - HNSW(Hierarchical Navigable Small World) 근사 알고리즘으로 sub-linear 시간 내 검색
- Wikipedia(2018년 12월 덤프)를 **100단어 단위 청크**로 분할 → 총 **2,100만 개 문서**
- 훈련 시 document encoder는 고정, **query encoder + BART generator만 fine-tuning**

---

## 5. Generator: BART

- **BART-large**: 400M 파라미터의 사전학습된 seq2seq Transformer
- 입력 구성: `[질문 x] + [검색된 문서 z]` 를 단순 concat해서 입력
- 디노이징(Denoising) 목적함수로 사전학습된 모델 — 다양한 생성 태스크에서 강점

---

## 6. 학습 방법

- Retriever와 Generator를 **End-to-End로 공동 학습**
- 어떤 문서를 검색해야 하는지 **직접적인 정답 레이블 없이** 학습 (약한 지도학습)
- 손실함수: 정답 y에 대한 **Negative Marginal Log-Likelihood** 최소화

```
Loss = -log p(y|x) = -log Σ p(z|x) · p(y|x, z)
```

---

## 7. 실험 및 결과

### 사용한 벤치마크
| 태스크 | 데이터셋 |
|--------|----------|
| Open-domain QA (추출형) | NaturalQuestions, TriviaQA, WebQuestions, CuratedTrec |
| Abstractive QA (생성형) | MS-MARCO |
| 질문 생성 | Jeopardy Question Generation |
| 사실 검증 | FEVER |

### 주요 결과

**Open-domain QA** — 4개 데이터셋 모두에서 당시 SOTA 달성

| 모델 | NaturalQuestions | TriviaQA | WebQuestions |
|------|-----------------|----------|--------------|
| T5-11B (Closed Book) | 34.5 | 50.1 | 37.4 |
| DPR (Open Book) | 41.5 | 57.9 | 41.1 |
| **RAG-Sequence** | **44.5** | **56.8** | **45.2** |

**Jeopardy 질문 생성 - 인간 평가 결과**

| 평가 항목 | BART가 더 좋음 | RAG가 더 좋음 |
|----------|--------------|--------------|
| 사실성 (Factuality) | 7.1% | **42.7%** |
| 구체성 (Specificity) | 16.8% | **37.4%** |

→ RAG가 BART 대비 훨씬 사실적이고 구체적인 답변 생성

---

## 8. 핵심 발견들

### 1. 생성형 접근이 추출형을 능가
- 추출형 모델은 문서에 답이 없으면 0점이지만
- RAG는 검색된 문서에 답이 없어도 **11.8% 정확도**로 정답 생성 가능 (파라미터 기억 활용)

### 2. Index Hot-Swapping (인덱스 교체)
- 재학습 없이 Wikipedia 인덱스만 교체해도 지식 업데이트 가능
- 2016년 인덱스 → 2016년 세계 리더 질문: **70% 정확**
- 2018년 인덱스 → 2018년 세계 리더 질문: **68% 정확**
- 인덱스 불일치 시 정확도 4~12%로 급락 → 인덱스가 지식의 핵심임을 증명

### 3. 학습된 Retriever가 BM25보다 우수
- 대부분의 태스크에서 Dense Retrieval > BM25 (키워드 기반)
- 단, FEVER(사실 검증)는 엔티티 중심이라 BM25도 경쟁력 있음

### 4. 생성 다양성
- RAG 생성물이 BART보다 **더 다양** (n-gram 다양성 지표)
- 별도의 다양성 디코딩 기법 없이도 달성

---

## 9. 한계 및 향후 과제 (Discussion)

- Retriever와 Generator를 처음부터 **Joint Pre-training** 하는 연구 필요
- 현재는 Wikipedia만 사용 → 다양한 외부 지식 소스 적용 가능성
- Wikipedia 자체의 편향성, 불완전성 문제 → 생성 콘텐츠 신뢰성 이슈

---

## 10. 논문의 의의

이 논문은 오늘날 모든 RAG 시스템의 **이론적 기반**이 된 논문이다.
현재 LangChain, LlamaIndex 등 프레임워크에서 구현하는 RAG 파이프라인은
사실상 이 논문의 아이디어를 실용화한 것이다.

```
논문의 RAG          →  현대 RAG 시스템
─────────────────────────────────────────────────
DPR (Bi-Encoder)    →  OpenAI / HuggingFace Embeddings
FAISS Index         →  Chroma, Pinecone, Qdrant 등 Vector DB
BART Generator      →  GPT-4, Claude 등 LLM
Wikipedia Dump      →  사내 문서, PDF, 웹페이지 등 커스텀 문서
```

---

## 참고

- 원논문: https://arxiv.org/abs/2005.11401
- HuggingFace 공식 구현: https://github.com/huggingface/transformers/tree/master/examples/rag
