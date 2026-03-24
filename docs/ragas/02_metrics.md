# Ragas 메트릭 상세 (Metrics)

---

## RAG 시스템 평가 메트릭

### 1. Faithfulness (충실도)

**측정 목적**: 생성된 답변이 검색된 컨텍스트에 **사실적으로 근거**하는지 평가

**작동 원리**:
1. 답변에서 모든 주장(claim)을 추출
2. 각 주장이 컨텍스트에서 지지되는지 LLM이 판단
3. `지지되는 주장 수 / 전체 주장 수` 로 점수 계산

**점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
scorer = Faithfulness(llm=llm)

result = await scorer.ascore(
    user_input="아인슈타인은 언제 태어났나요?",
    response="아인슈타인은 1879년 3월 20일 독일에서 태어났습니다.",
    retrieved_contexts=[
        "알베르트 아인슈타인(1879년 3월 14일 ~ 1955년)은 독일 출신의 이론물리학자다."
    ]
)
print(f"Faithfulness Score: {result.value}")
# → 0.5 (날짜 '20일'이 틀렸으므로 1/2 클레임만 지지됨)
```

**계산 예시:**
- 클레임 1: "독일에서 태어남" → 컨텍스트에 지지됨 ✓
- 클레임 2: "1879년 3월 20일" → 컨텍스트는 14일 → 지지 안 됨 ✗
- 최종 점수: 1/2 = **0.5**

**언제 낮아지는가?**
- LLM이 컨텍스트에 없는 내용을 만들어냈을 때 (환각)
- 컨텍스트의 내용을 왜곡했을 때

---

### 2. Response Relevancy (응답 관련성)

**측정 목적**: 생성된 답변이 사용자 질문과 **얼마나 관련 있는지** 평가

**작동 원리**:
1. 답변을 기반으로 여러 개의 가상 질문 생성
2. 생성된 가상 질문과 원래 질문 사이의 **코사인 유사도** 계산
3. 평균값이 최종 점수

**점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
result = await scorer.ascore(
    user_input="프랑스는 어디에 있고 수도는 어디인가요?",
    response="프랑스는 서유럽에 위치하며 수도는 파리입니다."
)
print(f"Answer Relevancy Score: {result.value}")
# → 약 0.92 (질문의 두 측면 모두 답변)
```

**수식:**
```
Answer Relevancy = (1/N) × Σ cosine_similarity(E_생성질문_i, E_원래질문)
```
N개의 역생성 질문과 원래 질문 사이의 평균 코사인 유사도

**언제 낮아지는가?**
- 답변이 질문과 무관한 내용을 포함할 때
- 답변이 너무 일반적이거나 포괄적일 때

---

### 3. Context Precision (컨텍스트 정밀도)

**측정 목적**: 검색된 컨텍스트 중 **실제로 유용한 컨텍스트의 비율** 평가

**작동 원리**:
- 검색된 각 컨텍스트가 정답 생성에 필요한지 LLM이 판단
- 관련 컨텍스트가 상위 순위에 있을수록 높은 점수

**점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

```python
from ragas.metrics.collections import ContextPrecision

scorer = ContextPrecision(llm=llm)
result = await scorer.ascore(
    user_input="에펠탑은 어디에 있나요?",
    reference="에펠탑은 파리에 있습니다.",
    retrieved_contexts=[
        "에펠탑은 프랑스 파리에 위치한 철제 격자 탑이다.",  # 관련 있음 ✓
        "파리는 패션의 도시로 유명하다.",                   # 관련 없음 ✗
        "에펠탑은 1889년에 완공되었다.",                    # 관련 있음 ✓
    ]
)
print(f"Context Precision Score: {result.value}")
```

**수식:**
```
Context Precision@K = Σ(Precision@k × vk) / 전체 관련 아이템 수
```
관련 청크가 상위 순위에 있을수록 점수가 높아짐

**언제 낮아지는가?**
- 관련 없는 문서가 많이 검색될 때
- 관련 문서가 낮은 순위에 있을 때

---

### 4. Context Recall (컨텍스트 재현율)

**측정 목적**: 정답을 생성하기 위해 **필요한 정보가 검색되었는지** 평가

**작동 원리**:
- 정답(reference)의 각 문장이 컨텍스트에서 지지되는지 확인
- `컨텍스트에서 지지되는 정답 문장 수 / 전체 정답 문장 수`

**점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

```python
from ragas.metrics.collections import ContextRecall

scorer = ContextRecall(llm=llm)
result = await scorer.ascore(
    user_input="에펠탑은 어디에 있나요?",
    retrieved_contexts=["파리는 프랑스의 수도입니다."],
    reference="에펠탑은 프랑스 파리에 위치해 있습니다."
)
print(f"Context Recall Score: {result.value}")
```

**수식:**
```
Context Recall = 컨텍스트에서 지지되는 정답 클레임 수 / 전체 정답 클레임 수
```

**언제 낮아지는가?**
- 리트리버가 관련 문서를 놓쳤을 때
- 청크 크기가 너무 작아 필요한 정보가 분산될 때

---

### 5. Context Entities Recall (컨텍스트 엔티티 재현율)

**측정 목적**: 정답에 있는 **핵심 엔티티(개체명)**가 컨텍스트에 얼마나 포함되었는지 평가

**작동 원리**:
- 정답에서 엔티티 추출 (사람 이름, 장소, 수치 등)
- 각 엔티티가 컨텍스트에 존재하는지 확인

```python
from ragas.metrics import context_entity_recall
```

---

### 6. Noise Sensitivity (노이즈 민감도)

**측정 목적**: 관련 없는(노이즈) 컨텍스트가 있을 때 LLM이 **얼마나 영향을 받는지** 평가

- 낮을수록 좋음 (노이즈에 강건한 시스템)

```python
from ragas.metrics import noise_sensitivity_relevant
```

---

## 자연어 평가 메트릭

### 7. Factual Correctness (사실적 정확성)

**측정 목적**: 응답의 사실적 내용이 **정답(reference)과 얼마나 일치**하는지 평가

- Precision, Recall, F1 형태로 결과 제공
- 정보의 정확성에 초점

```python
from ragas.metrics import factual_correctness
```

---

### 8. Semantic Similarity (의미적 유사도)

**측정 목적**: 응답과 정답 사이의 **의미적 유사도** 측정

- 임베딩 모델을 사용해 코사인 유사도 계산
- 표현이 다르더라도 의미가 같으면 높은 점수

```python
from ragas.metrics import semantic_similarity
```

---

## 에이전트/도구 평가 메트릭

### 9. Tool Call Accuracy (도구 호출 정확도)

**측정 목적**: 에이전트가 **올바른 도구를 올바른 인수로** 호출했는지 평가

```python
from ragas.metrics import tool_call_accuracy
```

### 10. Tool Call F1

도구 호출의 F1 점수 (Precision과 Recall의 조화 평균)

### 11. Agent Goal Accuracy (에이전트 목표 달성도)

**측정 목적**: 에이전트가 **최종 목표를 달성**했는지 평가

```python
from ragas.metrics import agent_goal_accuracy
```

### 12. Topic Adherence (주제 준수)

**측정 목적**: 에이전트가 지정된 **주제 범위 내에서만** 답변하는지 평가

---

## 범용 평가 메트릭

### 13. Aspect Critic (특정 측면 평가)

**측정 목적**: 사용자가 정의한 **특정 기준**으로 응답을 평가 (이진 평가)

```python
from ragas.metrics import AspectCritic

# 예시: 유해성 평가
harmfulness = AspectCritic(
    name="harmfulness",
    definition="응답이 해롭거나 부적절한 내용을 포함하는가?"
)
```

### 14. Simple Criteria Scoring (단순 기준 점수)

사용자 정의 기준으로 0~1 범위의 점수 부여

### 15. Rubrics-Based Scoring (기준표 기반 평가)

**측정 목적**: 세분화된 **기준표(Rubric)**에 따라 다단계 점수 부여

```python
from ragas.metrics import RubricsScore

rubric = RubricsScore(
    name="quality_score",
    rubrics={
        "score1_description": "전혀 도움이 되지 않음",
        "score2_description": "약간 도움이 됨",
        "score3_description": "보통 수준",
        "score4_description": "도움이 됨",
        "score5_description": "매우 도움이 됨"
    }
)
```

### 16. Instance-Specific Rubrics Scoring

각 샘플별로 다른 기준표를 적용하는 메트릭

---

## 전통적 NLP 메트릭

### 17. BLEU Score

기계 번역 품질 평가에서 유래한 n-gram 기반 메트릭

```python
from ragas.metrics import BleuScore
```

### 18. ROUGE Score

요약 평가에 주로 사용되는 재현율 기반 메트릭

```python
from ragas.metrics import RougeScore
```

### 19. CHRF Score

문자 n-gram F-score 기반 메트릭

### 20. Exact Match

응답이 정답과 정확히 일치하는지 확인

### 21. String Presence

응답에 특정 문자열이 포함되는지 확인

---

## SQL 평가 메트릭

### 22. SQL Query Equivalence

두 SQL 쿼리가 동등한지 구조적으로 평가

### 23. Execution-based Datacompy Score

실제 SQL 실행 결과를 비교하는 평가

---

## 요약 평가 메트릭

### 24. Summarization Score

요약의 품질을 평가하는 메트릭

---

## NVIDIA 메트릭

| 메트릭 | 설명 |
|--------|------|
| Answer Accuracy | 답변의 정확도 |
| Context Relevance | 컨텍스트 관련성 |
| Response Groundedness | 응답의 근거 충실도 |

```python
from ragas.metrics.nvidia import AnswerAccuracy, ContextRelevance, ResponseGroundedness
```

---

## 메트릭 클래스 구조

```
Metric (추상 기반)
├── MetricWithLLM (LLM 평가 기반)
│   ├── SingleTurnMetric (단일 턴)
│   └── MultiTurnMetric (멀티 턴)
└── SimpleBaseMetric
    └── SimpleLLMMetric (저장/로드 + 정렬 지원)
```

### 핵심 메서드

| 메서드 | 설명 |
|--------|------|
| `init()` | 메트릭 초기화 (LLM 설정 등) |
| `score()` | 동기 방식 점수 계산 |
| `ascore()` | 비동기 방식 점수 계산 |
| `single_turn_score()` | 단일 턴 동기 평가 |
| `multi_turn_score()` | 멀티 턴 동기 평가 |

---

## 메트릭 선택 가이드

```
평가하고 싶은 것이 무엇인가?
│
├── 검색 성능 → Context Precision + Context Recall
│
├── 생성 품질
│   ├── 사실성/환각 방지 → Faithfulness
│   ├── 질문 적합성 → Response Relevancy
│   └── 정보 정확성 → Factual Correctness
│
├── 전반적인 RAG 품질
│   └── Faithfulness + Response Relevancy + Context Precision + Context Recall
│
├── 에이전트 평가 → Tool Call Accuracy + Agent Goal Accuracy
│
└── 커스텀 기준 → Aspect Critic 또는 Rubrics-Based Scoring
```
