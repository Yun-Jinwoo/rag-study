# RAG 평가 파이프라인 학습 로드맵

---

## 목표 시스템 구조

```
[RAG 파이프라인]              [평가 파이프라인]
문서 전처리 (청킹, 임베딩)  →  테스트셋 구성
벡터 데이터베이스           →  Ragas 메트릭 측정
리트리버                    →  결과 분석
LLM 기반 답변 생성          →  파라미터 튜닝
```

---

## 단계별 학습 순서

### Phase 1: 기초 이해

**목표**: RAG의 개념과 기본 구현 이해

- RAG 개념 — 검색(Retrieval)과 생성(Generation)의 결합 원리
- 임베딩(Embedding) 동작 원리
- 벡터 데이터베이스 (FAISS, Chroma, Pinecone 등)
- 청킹(Chunking) 전략 (크기, 겹침, 분할 방법)
- LangChain 기초 — 간단한 RAG 파이프라인 예제 실행

```python
# 목표: 가장 간단한 RAG 파이프라인 구축
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 문서 로드
# 2. 청킹
# 3. 임베딩 & 벡터 스토어
# 4. 리트리버 설정
# 5. QnA 체인 구성
# 6. 질문하기
```

---

### Phase 2: Ragas 평가 시스템

**목표**: Ragas를 이용한 RAG 평가 파이프라인 구축

**핵심 메트릭 (우선순위 순):**

1. `Faithfulness` — 답변이 문서에 근거하는가?
2. `Response Relevancy` — 답변이 질문에 적합한가?
3. `Context Precision` — 검색 결과가 정확한가?
4. `Context Recall` — 필요한 내용을 모두 검색했는가?

**학습 내용:**
- 평가 데이터셋 구성 (수동으로 질문-정답 쌍 작성)
- Ragas `TestsetGenerator`로 합성 데이터셋 생성
- `evaluate()` 함수 사용법
- 점수가 낮을 때 원인 파악 방법

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

dataset = [...]  # Phase 1 RAG 실행 결과

result = evaluate(dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
])
print(result)
```

---

### Phase 3: 파이프라인 고도화

**목표**: 실험 기반 개선 루프 구축

- `TestsetGenerator` 활용한 테스트셋 생성 자동화
- 다양한 질문 유형 (SingleHop, MultiHop)
- 기준선(Baseline) 설정 후 변수 하나씩 바꾸며 효과 측정
- 청킹 전략 비교 실험

**파라미터 튜닝 대상:**

| 변수 | 실험 범위 | 관련 메트릭 |
|------|-----------|------------|
| 청크 크기 | 512 / 1024 / 2048 | Context Recall |
| 검색 문서 수 (top-k) | 3 / 5 / 10 | Faithfulness |
| 임베딩 모델 | OpenAI / HuggingFace | 전체 |
| 리랭킹 적용 여부 | 없음 / Cohere Rerank | Context Precision |

---

## 기술 스택 정리

### 필수

| 분류 | 기술 | 용도 |
|------|------|------|
| RAG 프레임워크 | LangChain | 전체 파이프라인 |
| LLM | OpenAI GPT-4o-mini | 답변 생성 |
| 임베딩 | OpenAI text-embedding-3-small | 벡터화 |
| 벡터 DB | FAISS (로컬) 또는 Chroma | 문서 저장 |
| 평가 | Ragas | 메트릭 측정 |
| 언어 | Python 3.10+ | — |

### 선택 (고도화 시)

| 분류 | 기술 | 용도 |
|------|------|------|
| 관측 | LangSmith | 평가 트레이싱 |
| 로컬 LLM | Ollama | 비용 절감 |
| 벡터 DB | Pinecone / Qdrant | 클라우드 확장 |
| 리랭킹 | Cohere Rerank | 검색 품질 향상 |

---

## 참고 자료

| 자료 | 링크 |
|------|------|
| Ragas 공식 문서 | https://docs.ragas.io/en/stable/ |
| LangChain 문서 | https://docs.langchain.com/ |
| LangGraph 문서 | https://docs.langchain.com/oss/python/langgraph/overview |
| LangSmith | https://smith.langchain.com/ |
| FAISS | https://github.com/facebookresearch/faiss |

---

## 용어 정리

| 용어 | 설명 |
|------|------|
| RAG | Retrieval-Augmented Generation, 검색 증강 생성 |
| Chunk | 문서를 분할한 조각 |
| Embedding | 텍스트를 숫자 벡터로 변환 |
| Vector Store | 임베딩 벡터를 저장하는 데이터베이스 |
| Retriever | 질문과 유사한 청크를 검색하는 모듈 |
| Hallucination | LLM이 사실이 아닌 내용을 만들어내는 현상 |
| Faithfulness | 답변이 컨텍스트에 근거하는 정도 |
| Context Precision | 검색된 문서 중 유용한 문서의 비율 |
| Context Recall | 필요한 정보가 검색된 비율 |
| Grounding | 답변이 실제 문서에 근거를 두는 것 |
| LLM-as-a-Judge | LLM이 평가자 역할을 하는 방식 |
| Synthetic Testset | LLM으로 자동 생성한 테스트 데이터셋 |
| SingleHop | 단일 문서에서 답변 가능한 질문 |
| MultiHop | 여러 문서를 종합해야 답변 가능한 질문 |
