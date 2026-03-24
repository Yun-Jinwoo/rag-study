# Ragas API 레퍼런스 (API Reference)

---

## LLM 관련 클래스

### BaseRagasLLM

모든 Ragas LLM 래퍼의 추상 기반 클래스

```python
from ragas.llms.base import BaseRagasLLM

# 주요 특징
# - 자동 온도(temperature) 관리
# - Usage 이벤트 추적
# - 동기/비동기 메서드 모두 지원
```

### LiteLLMStructuredLLM

100+ LLM 제공자 지원

```python
from ragas.llms import LiteLLMStructuredLLM

llm = LiteLLMStructuredLLM(
    model="openai/gpt-4o",        # OpenAI
    # model="ollama/llama3.2",    # Ollama
    # model="groq/llama-3.1-70b", # Groq
    # model="anthropic/claude-3", # Anthropic
)
```

### llm_factory()

LLM 인스턴스 자동 생성 팩토리 함수

```python
from ragas.llms import llm_factory

# 환경 변수 기반 자동 감지
llm = llm_factory()
```

---

## 임베딩 관련 클래스

### 임베딩 클래스 계층

```
BaseRagasEmbedding (추상 기반)
└── BaseRagasEmbeddings (RunConfig 관리 추가)
    ├── OpenAIEmbeddings
    ├── GoogleEmbeddings (Vertex AI + Gemini)
    ├── HuggingFaceEmbeddings
    ├── LiteLLMEmbeddings
    └── HaystackEmbeddingsWrapper
```

### 주요 유틸리티 함수

```python
from ragas.embeddings.utils import (
    batch_texts,          # 텍스트 배치 처리
    get_optimal_batch_size,  # 제공자별 최적 배치 크기 반환
    validate_texts        # 텍스트 유효성 검사
)
```

---

## 실행 설정 (RunConfig)

```python
from ragas import RunConfig

config = RunConfig(
    timeout=180,          # API 요청 타임아웃 (초), 기본값: 180
    max_retries=10,       # 최대 재시도 횟수, 기본값: 10
    max_wait=60,          # 재시도 간 최대 대기 시간 (초), 기본값: 60
    max_workers=16,       # 병렬 처리 워커 수, 기본값: 16
    log_tenacity=False,   # 재시도 로깅 여부
    seed=42               # 랜덤 시드, 기본값: 42
)

# evaluate()에 적용
result = evaluate(dataset, metrics=[...], run_config=config)
```

### 재시도 로직

```python
from ragas.run_config import add_retry, add_async_retry

# Tenacity 기반 지수 백오프 재시도 자동 적용
# - 네트워크 오류 시 자동 재시도
# - Rate Limit 오류 시 대기 후 재시도
```

---

## 캐싱 (Cache)

성능 향상을 위한 API 응답 캐싱 (최대 **60배 속도 향상**)

```python
from ragas.cache import DiskCacheBackend

# 디스크 캐시 설정
cache = DiskCacheBackend(cache_dir=".ragas_cache")

# evaluate()에 캐시 적용
result = evaluate(
    dataset,
    metrics=[...],
    # cache_backend=cache  # 향후 지원 예정
)
```

### cacher 데코레이터

```python
from ragas.cache import cacher

@cacher(cache_backend=cache)
async def expensive_llm_call(prompt: str) -> str:
    # 첫 호출 후 캐시에 저장됨
    return await llm.generate(prompt)
```

---

## Executor (비동기 실행기)

```python
from ragas.executor import Executor

executor = Executor(
    desc="평가 진행 중",        # 진행 표시줄 설명
    show_progress=True,         # 진행 표시줄 표시
    keep_progress_bar=False,    # 완료 후 표시줄 유지 여부
    batch_size=10,              # 배치 크기
    raise_exceptions=True,      # 예외 발생 시 즉시 중단
    run_config=run_config       # RunConfig 설정
)

# 작업 제출
executor.submit(async_function, *args)

# 결과 수집
results = await executor.aresults()

# 작업 취소
executor.cancel()
```

### run_async_batch()

```python
from ragas.executor import run_async_batch

# 여러 비동기 작업 병렬 실행
results = await run_async_batch(
    jobs=[async_task1, async_task2, ...],
    batch_size=10
)
```

---

## 토크나이저 (Tokenizer)

```python
from ragas.tokenizers import get_tokenizer, TiktokenWrapper, HuggingFaceTokenizer

# 기본 토크나이저 (OpenAI tiktoken, o200k_base 인코딩)
tokenizer = get_tokenizer()

# 직접 지정
tiktoken_tok = TiktokenWrapper(encoding="cl100k_base")
hf_tok = HuggingFaceTokenizer(model_name="bert-base-uncased")

# 주요 메서드
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
count = tokenizer.count_tokens("Hello, world!")  # = 4
```

---

## EvaluationDataset

```python
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample

# 단일 턴 샘플 생성
sample = SingleTurnSample(
    user_input="RAG란 무엇인가?",
    retrieved_contexts=["RAG는 검색 증강 생성 기술로..."],
    response="RAG는 검색과 생성을 결합한 기술입니다.",
    reference="RAG(Retrieval-Augmented Generation)는..."
)

# 데이터셋 생성
dataset = EvaluationDataset(samples=[sample1, sample2, ...])

# 저장/로드
dataset.to_csv("eval_data.csv")
dataset = EvaluationDataset.from_csv("eval_data.csv")

# pandas DataFrame으로 변환
df = dataset.to_pandas()
```

### 멀티턴 샘플

```python
from ragas.dataset_schema import MultiTurnSample, HumanMessage, AIMessage

sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="RAG가 무엇인가요?"),
        AIMessage(content="RAG는 검색 증강 생성입니다."),
        HumanMessage(content="어떤 장점이 있나요?"),
    ],
    reference="RAG의 주요 장점은 환각 감소와 최신 정보 반영입니다."
)
```

---

## evaluate() 함수

```python
from ragas import evaluate

result = evaluate(
    dataset=eval_dataset,           # EvaluationDataset
    metrics=[faithfulness, ...],    # 평가할 메트릭 목록
    llm=llm,                        # LLM (선택적)
    embeddings=embeddings,          # 임베딩 모델 (선택적)
    run_config=run_config,          # 실행 설정 (선택적)
    batch_size=15,                  # 배치 크기
    show_progress=True,             # 진행 표시
    raise_exceptions=False,         # 예외 처리 방식
)

# 결과 확인
print(result)                        # 평균 점수 출력
df = result.to_pandas()             # 상세 결과 DataFrame
result.upload()                      # 관측 플랫폼 업로드
```

---

## 커스텀 메트릭 작성

### 데코레이터 방식 (간단)

```python
from ragas.metrics.base import metric

@metric
def my_custom_metric(sample):
    """커스텀 평가 로직"""
    # sample.user_input
    # sample.response
    # sample.retrieved_contexts
    score = ...  # 0.0 ~ 1.0
    return score
```

### 클래스 방식 (고급)

```python
from ragas.metrics.base import SingleTurnMetric, MetricWithLLM
from ragas.dataset_schema import SingleTurnSample
from dataclasses import dataclass

@dataclass
class MyMetric(MetricWithLLM, SingleTurnMetric):
    name: str = "my_metric"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        # LLM을 사용한 평가 로직
        prompt = f"Evaluate: {sample.response}"
        result = await self.llm.generate(prompt)
        return parse_score(result)

    def init(self, run_config):
        # 초기화 로직
        pass
```

---

## 프롬프트 커스터마이징

```python
from ragas.metrics import Faithfulness

# 메트릭 프롬프트 수정
faithfulness_metric = Faithfulness()

# 기존 프롬프트 확인
print(faithfulness_metric.statement_prompt.instruction)

# 프롬프트 수정 (한국어 등)
faithfulness_metric.statement_prompt.instruction = "다음 응답에서 모든 진술을 추출하세요..."
```
