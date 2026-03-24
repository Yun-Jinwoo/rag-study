# RAG 기반 QnA 시스템 & 신뢰성 평가 파이프라인 - 학습 계획

> 캡스톤 프로젝트: "RAG 기반 QnA 시스템 및 응답 신뢰성 평가 파이프라인 개발"

---

## 프로젝트 이해

### 핵심 목표

```
[RAG 기반 QnA 시스템] + [응답 신뢰성 평가 파이프라인]
         ↓                            ↓
  질문에 대해 외부               Ragas 메트릭으로
  문서를 검색하고               응답의 품질/신뢰도를
  답변을 생성                   자동으로 측정
```

### 최종 결과물 구성

```
프로젝트
├── RAG 파이프라인
│   ├── 문서 전처리 (청킹, 임베딩)
│   ├── 벡터 데이터베이스
│   ├── 리트리버
│   └── LLM 기반 답변 생성
│
├── 평가 파이프라인
│   ├── 테스트셋 (합성 또는 수동)
│   ├── Ragas 메트릭 측정
│   └── 결과 시각화/리포트
│
└── 실험 및 개선
    ├── 파라미터 튜닝
    ├── A/B 테스트
    └── 성능 개선 보고서
```

---

## 단계별 학습 계획

### Phase 1: 기초 이해 (1~2주차)

**목표**: RAG의 개념과 기본 구현 이해

#### 주요 학습 내용

- [ ] **RAG 개념** 이해
  - 검색(Retrieval)과 생성(Generation)의 결합 원리
  - 왜 RAG가 필요한가? (환각 문제, 최신 정보 한계)
  - RAG의 각 구성요소 역할

- [ ] **핵심 기술 스택** 파악
  - 임베딩(Embedding)이란? 어떻게 동작하는가?
  - 벡터 데이터베이스란? (FAISS, Chroma, Pinecone 등)
  - 청킹(Chunking) 전략 (크기, 겹침, 분할 방법)

- [ ] **LangChain 기초** (권장 프레임워크)
  - LangChain 문서 훑어보기
  - 간단한 RAG 파이프라인 예제 실행

#### 실습 과제

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

### Phase 2: Ragas 평가 시스템 (3~4주차)

**목표**: Ragas를 이용한 RAG 평가 파이프라인 구축

#### 주요 학습 내용

- [ ] **핵심 메트릭 이해** (우선순위 순)
  1. Faithfulness - 답변이 문서에 근거하는가?
  2. Response Relevancy - 답변이 질문에 적합한가?
  3. Context Precision - 검색 결과가 정확한가?
  4. Context Recall - 필요한 내용을 모두 검색했는가?

- [ ] **평가 데이터셋 구성**
  - 수동으로 질문-정답 쌍 작성 (소규모)
  - Ragas TestsetGenerator로 합성 데이터셋 생성

- [ ] **평가 실행 및 결과 해석**
  - `evaluate()` 함수 사용법
  - 점수가 낮을 때 원인 파악 방법

#### 실습 과제

```python
# 목표: Phase 1 RAG 파이프라인에 Ragas 평가 연결
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Phase 1에서 만든 RAG 실행 후 결과를 dataset으로 구성
dataset = [...]

result = evaluate(dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
])

print(result)
# 목표: 각 점수가 어떤 의미인지 팀원과 함께 분석
```

---

### Phase 3: 파이프라인 고도화 (5~6주차)

**목표**: 실험 기반 개선 루프 구축

#### 주요 학습 내용

- [ ] **테스트셋 생성 자동화**
  - TestsetGenerator 활용
  - 다양한 질문 유형 (SingleHop, MultiHop)
  - 한국어 테스트셋 생성

- [ ] **실험 설계**
  - 기준선(Baseline) 설정
  - 변수 하나씩 바꾸며 효과 측정
  - 청킹 전략 비교 실험

- [ ] **파라미터 튜닝**
  - 청크 크기 (512 vs 1024 vs 2048)
  - 검색 문서 수 (top-k: 3 vs 5 vs 10)
  - 임베딩 모델 비교

#### 실험 계획표 (예시)

| 실험 | 변경 사항 | 기대 효과 | 평가 메트릭 |
|------|-----------|-----------|------------|
| Exp-01 | 청크 크기 512 → 1024 | Context Recall 향상 | Context Recall |
| Exp-02 | top-k 3 → 5 | Faithfulness 향상 | Faithfulness |
| Exp-03 | 리랭킹 추가 | Context Precision 향상 | Context Precision |

---

### Phase 4: 시스템 완성 및 발표 준비 (7~8주차)

**목표**: 최종 시스템 통합 및 결과 정리

#### 주요 학습 내용

- [ ] **결과 시각화**
  - 실험별 메트릭 비교 차트
  - 개선 전/후 성능 비교

- [ ] **최종 보고서 작성**
  - 사용한 기술 스택 정리
  - 실험 결과 및 인사이트
  - 개선 방향 및 한계

- [ ] **데모 준비**
  - 실시간 QnA 시연
  - 평가 파이프라인 시연

---

## 기술 스택 권장사항

### 필수 기술

| 분류 | 기술 | 용도 |
|------|------|------|
| RAG 프레임워크 | LangChain 또는 LlamaIndex | 전체 파이프라인 |
| LLM | OpenAI GPT-4o-mini | 답변 생성 |
| 임베딩 | OpenAI text-embedding-3-small | 벡터화 |
| 벡터 DB | FAISS (로컬) 또는 Chroma | 문서 저장 |
| 평가 | Ragas | 메트릭 측정 |
| 언어 | Python 3.10+ | - |

### 선택 기술 (고도화)

| 분류 | 기술 | 용도 |
|------|------|------|
| 관측 | LangSmith | 평가 트레이싱 |
| 로컬 LLM | Ollama | 비용 절감 |
| 벡터 DB | Pinecone / Qdrant | 클라우드 운영 |
| 리랭킹 | Cohere Rerank | 검색 품질 향상 |

---

## 팀 역할 분담 제안 (5인 팀)

```
팀원 A: RAG 파이프라인 구축 (인덱싱)
  - 문서 로딩 및 전처리 (Document Loaders)
  - 청킹 전략 구현 (Text Splitters)
  - 임베딩 모델 선택 및 적용
  - 벡터 스토어 구성 (Chroma / FAISS)
  참고 파일: langchain/02_rag_pipeline.md

팀원 B: RAG 파이프라인 구축 (검색 + 생성)
  - 리트리버 설정 및 튜닝 (top-k, 유사도 임계값)
  - LLM 에이전트/체인 구성
  - 시스템 프롬프트 설계
  - Indirect Prompt Injection 방어
  참고 파일: langchain/03_agents_tools.md, langchain/01_models.md

팀원 C: 평가 파이프라인 구축
  - Ragas 메트릭 설정 및 실행
  - 테스트셋 생성/관리 (수동 + TestsetGenerator)
  - 평가 결과 수집 및 저장
  참고 파일: docs/02_metrics.md, docs/03_testset_generation.md

팀원 D: 실험 설계 및 결과 분석
  - 실험 계획 수립 (변수 정의, 기준선 설정)
  - 파라미터 튜닝 (청크 크기, top-k, 임베딩 모델 등)
  - A/B 테스트 관리 및 결과 비교 분석
  - 실험 결과 시각화 (차트, 표)
  참고 파일: docs/06_learning_plan.md (실험 계획표)

팀원 E: 통합 및 인프라
  - 전체 파이프라인 통합 및 연결
  - LangSmith 트레이싱 설정
  - 최종 데모 환경 구성
  - 발표 자료 및 보고서 작성
  참고 파일: langchain/06_langsmith.md, langchain/07_study_guide.md
```

---

## 이번 주 숙제 (개인 학습) 체크리스트

- [ ] RAG 개념 이해: 검색 + 생성이 어떻게 작동하는지 설명할 수 있다
- [ ] Ragas 문서 요약 파일 읽기 (이 폴더의 md 파일들)
- [ ] Ragas 핵심 4개 메트릭 설명할 수 있다
  - Faithfulness, Response Relevancy, Context Precision, Context Recall
- [ ] Python 환경에서 `pip install ragas langchain langchain-openai` 실행해보기
- [ ] Ragas 공식 예제 코드 한 번 돌려보기

---

## 참고 자료

### 이 폴더의 정리 문서

**Ragas 관련:**
| 파일 | 내용 |
|------|------|
| `docs/00_overview.md` | Ragas 개요 및 핵심 기능 |
| `docs/01_core_concepts.md` | 실험, 데이터셋, 메트릭 개념 |
| `docs/02_metrics.md` | 전체 메트릭 상세 및 코드 예제 |
| `docs/03_testset_generation.md` | 테스트셋 자동 생성 |
| `docs/04_integrations.md` | LangChain, LlamaIndex 연동 |
| `docs/05_api_reference.md` | API 레퍼런스 |

**LangChain 관련:**
| 파일 | 내용 |
|------|------|
| `docs/langchain/00_overview.md` | LangChain 개요, 생태계, 설치 |
| `docs/langchain/01_models.md` | Chat Models, 모델 초기화/호출 |
| `docs/langchain/02_rag_pipeline.md` | RAG 전체 파이프라인 구현 |
| `docs/langchain/03_agents_tools.md` | 에이전트, 도구 정의 및 사용 |
| `docs/langchain/04_multi_agent.md` | 멀티 에이전트 패턴 5가지 |
| `docs/langchain/05_langgraph.md` | LangGraph 저수준 오케스트레이션 |
| `docs/langchain/06_langsmith.md` | 트레이싱·평가·배포 플랫폼 |
| `docs/langchain/07_study_guide.md` | 통합 학습 가이드 + 빠른 시작 |

### 외부 공식 문서

| 자료 | 링크 | 설명 |
|------|------|------|
| Ragas 공식 문서 | https://docs.ragas.io/en/stable/ | 메인 레퍼런스 |
| LangChain 문서 | https://docs.langchain.com/ | RAG 프레임워크 |
| LangGraph 문서 | https://docs.langchain.com/oss/python/langgraph/overview | 고급 오케스트레이션 |
| LangSmith | https://smith.langchain.com/ | 트레이싱/평가 플랫폼 |
| LlamaIndex 문서 | https://docs.llamaindex.ai/ | 대안 프레임워크 |
| FAISS | https://github.com/facebookresearch/faiss | 로컬 벡터 DB |

---

## 자주 나오는 용어 정리

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
