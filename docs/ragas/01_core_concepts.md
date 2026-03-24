# Ragas 핵심 개념 (Core Concepts)

---

## 1. RAG (Retrieval-Augmented Generation) 이란?

RAG는 **검색(Retrieval)** 과 **생성(Generation)** 을 결합한 기술이다.

```
사용자 질문
    ↓
[Retriever] 문서 데이터베이스에서 관련 컨텍스트 검색
    ↓
[Generator] LLM이 컨텍스트를 기반으로 답변 생성
    ↓
최종 응답
```

### RAG의 구성 요소

| 구성 요소 | 역할 |
|-----------|------|
| **문서 저장소** | 원본 지식 베이스 (PDF, 웹페이지, DB 등) |
| **청킹 (Chunking)** | 문서를 적절한 크기의 조각으로 분할 |
| **임베딩 모델** | 텍스트를 벡터로 변환 |
| **벡터 데이터베이스** | 임베딩 벡터 저장 및 유사도 검색 |
| **리트리버 (Retriever)** | 질문과 관련된 청크 검색 |
| **LLM** | 검색된 컨텍스트 기반 답변 생성 |

### RAG의 장점

- **환각(Hallucination) 감소**: 실제 문서를 기반으로 답변 생성
- **최신 정보 반영**: 외부 지식 베이스 업데이트 가능
- **출처 추적 가능**: 어떤 문서를 기반으로 답했는지 확인 가능

---

## 2. 데이터셋 (Datasets)

### 데이터셋의 구성 요소

```python
{
    "id": "고유 식별자",                     # 선택
    "user_input": "사용자 질문",              # 필수
    "retrieved_contexts": ["컨텍스트1", ...], # RAG 평가 시 필수
    "response": "시스템 응답",               # 필수
    "reference": "정답/기대 응답",            # 일부 메트릭에 필요
    "metadata": { ... }                      # 선택
}
```

### EvaluationDataset 클래스

```python
from ragas import EvaluationDataset

dataset = EvaluationDataset(samples=[...])

# CSV로 저장
dataset.to_csv("eval_dataset.csv")

# CSV에서 로드
dataset = EvaluationDataset.from_csv("eval_dataset.csv")
```

### 데이터셋 모범 사례

1. **대표성**: 실제 사용 패턴을 반영하는 샘플 구성
2. **다양성**: 다양한 질문 유형과 난이도 포함
3. **균형**: 쉬운/어려운 케이스의 균형 유지
4. **메타데이터 풍부화**: 분석을 위한 충분한 메타데이터 포함
5. **버전 관리**: 재현성을 위한 버전 추적

### 저장 옵션

| 옵션 | 설명 |
|------|------|
| 로컬 CSV | 기본 파일 저장 |
| Google Drive | 실험적 클라우드 지원 |

---

## 3. 메트릭 (Metrics) 개요

Ragas 메트릭은 **LLM이 평가자 역할**을 하는 방식으로 동작한다 (LLM-as-a-Judge).

### 메트릭 카테고리

```
Ragas 메트릭
├── RAG 시스템 평가
│   ├── Context Precision    (컨텍스트 정밀도)
│   ├── Context Recall       (컨텍스트 재현율)
│   ├── Response Relevancy   (응답 관련성)
│   ├── Faithfulness         (충실도/사실성)
│   └── Noise Sensitivity    (노이즈 민감도)
│
├── 에이전트/도구 평가
│   ├── Tool Call Accuracy   (도구 호출 정확도)
│   ├── Agent Goal Accuracy  (에이전트 목표 달성도)
│   └── Topic Adherence      (주제 준수)
│
├── 자연어 평가
│   ├── Factual Correctness  (사실적 정확성)
│   └── Semantic Similarity  (의미적 유사도)
│
├── 전통적 NLP 메트릭
│   ├── BLEU Score
│   ├── ROUGE Score
│   ├── CHRF Score
│   └── Exact Match
│
└── 범용 평가
    ├── Aspect Critic        (특정 측면 평가)
    ├── Rubrics-Based Scoring (기준표 기반 평가)
    └── Simple Criteria Scoring
```

### 메트릭 타입

| 타입 | 설명 |
|------|------|
| `SINGLE_TURN` | 단일 질문-응답 쌍 평가 |
| `MULTI_TURN` | 멀티턴 대화 평가 |

---

## 4. 실험 (Experimentation)

실험은 **변경 사항의 효과를 측정**하기 위한 Ragas의 핵심 개념이다.

### 실험 워크플로우

```python
from ragas import evaluate

# 기준선(Baseline) 평가
baseline_result = evaluate(dataset, metrics=[...], llm=baseline_llm)

# 개선 버전 평가
improved_result = evaluate(dataset, metrics=[...], llm=improved_llm)

# 결과 비교
print(baseline_result.to_pandas())
print(improved_result.to_pandas())
```

### 실험에서 추적해야 할 것

- 프롬프트 변경 전/후 성능
- 다른 LLM 모델 비교
- 청킹 전략 변경 효과
- 임베딩 모델 변경 효과
- 리트리버 파라미터 조정 효과

---

## 5. 테스트셋 생성 (Testset Generation)

### 왜 합성 테스트셋이 필요한가?

- 수동 데이터셋 구축은 **시간이 많이 소요**됨
- 실제 데이터가 부족한 경우
- 다양한 시나리오를 **체계적으로** 커버해야 할 때
- 데이터 드리프트에 대응하는 **지속적인 업데이트** 필요

### 이상적인 테스트셋 요건

1. 고품질 데이터 샘플
2. 다양한 실제 시나리오 커버
3. 통계적으로 유의미한 샘플 수
4. 지속적인 업데이트

---

## 6. RunConfig (실행 설정)

```python
from ragas import RunConfig

run_config = RunConfig(
    timeout=180,      # 요청 타임아웃 (초)
    max_retries=10,   # 최대 재시도 횟수
    max_wait=60,      # 최대 대기 시간 (초)
    max_workers=16,   # 최대 병렬 워커 수
    seed=42           # 재현성을 위한 랜덤 시드
)
```

---

## 정리: RAG 시스템 평가의 핵심 질문들

| 질문 | 관련 메트릭 |
|------|------------|
| 관련 문서를 잘 가져왔는가? | Context Precision, Context Recall |
| 가져온 문서가 실제로 유용한가? | Noise Sensitivity |
| 답변이 문서 내용에 충실한가? | Faithfulness |
| 답변이 질문에 적합한가? | Response Relevancy |
| 답변이 사실적으로 정확한가? | Factual Correctness |
