# Ragas 통합 (Integrations)

---

## 지원 프레임워크 개요

```
Ragas 통합 생태계
│
├── RAG 프레임워크
│   ├── LangChain
│   ├── LlamaIndex
│   ├── Haystack
│   ├── Griptape
│   └── R2R
│
├── 에이전트 프레임워크
│   ├── LlamaStack (Meta)
│   ├── Swarm (OpenAI)
│   └── Amazon Bedrock Agents
│
├── LLM 제공자
│   ├── OpenAI
│   ├── Google Gemini
│   ├── Amazon Bedrock
│   ├── HuggingFace
│   └── LiteLLM (100+ 제공자)
│
└── 관측 플랫폼
    ├── LangSmith
    └── Arize Phoenix
```

---

## LangChain 통합

### 설치

```bash
pip install ragas langchain langchain-openai
```

### RAG 평가 예시

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.integrations.langchain import EvaluatorChain

# LangChain RAG 파이프라인 구성
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# ... vectorstore 및 retriever 설정 ...

# Ragas 평가
result = evaluate(
    dataset=eval_dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness()
    ]
)

# 예시 결과
# context_recall: 1.0
# faithfulness: 0.9
# factual_correctness: 0.926
print(result)
```

---

## Haystack 통합

### 파이프라인 구성

```python
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBEMRetriever
from haystack.components.generators import OpenAIGenerator
from ragas.integrations.haystack import RagasEvaluator

# 파이프라인 조립
pipeline = Pipeline()
pipeline.add_component("text_embedder", ...)
pipeline.add_component("retriever", InMemoryBEMRetriever(...))
pipeline.add_component("prompt_builder", ...)
pipeline.add_component("llm", OpenAIGenerator(...))
pipeline.add_component("answer_builder", ...)
pipeline.add_component("ragas_evaluator", RagasEvaluator(
    metrics=[answer_relevancy, context_precision, faithfulness]
))

# 예시 결과
# answer_relevancy: 0.9782
# context_precision: 1.0
# faithfulness: 1.0
```

---

## LlamaIndex 통합

### RAG 평가

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from ragas.integrations.llama_index import evaluate

# 인덱스 구성
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Ragas 평가
result = evaluate(
    query_engine=query_engine,
    eval_dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## LLM 제공자 통합

### OpenAI (기본)

```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
```

### Google Gemini

```python
# 권장: 새로운 google-genai SDK
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-2.0-flash")
)

# 임베딩은 자동으로 Google 임베딩으로 매칭됨
```

**지원 Gemini 모델:**
- `gemini-2.0-flash` (권장 - 속도/효율성)
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-1.0-pro`

### LiteLLM (100+ 제공자)

```python
from ragas.llms import LiteLLMStructuredLLM

# Ollama (로컬 모델)
llm = LiteLLMStructuredLLM(model="ollama/llama3.2")

# Groq
llm = LiteLLMStructuredLLM(model="groq/llama-3.1-70b-versatile")

# Anthropic Claude
llm = LiteLLMStructuredLLM(model="anthropic/claude-3-5-sonnet-20241022")
```

### Amazon Bedrock

```python
from ragas.llms import LangchainLLMWrapper
from langchain_aws import ChatBedrock

llm = LangchainLLMWrapper(
    ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1"
    )
)
```

### HuggingFace

```python
from ragas.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## 임베딩 모델별 최적 배치 크기

| 제공자 | 최적 배치 크기 |
|--------|---------------|
| OpenAI | 100 |
| Cohere | 96 |
| Google | 5 |
| HuggingFace | 32 |

---

## 관측 플랫폼 통합

### LangSmith

```python
import os

# 환경 변수 설정만으로 자동 추적
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "ragas-evaluation"

# 이후 evaluate() 호출 시 자동으로 LangSmith에 추적됨
result = evaluate(dataset, metrics=[...])
```

### Arize Phoenix

```python
import phoenix as px
from phoenix.otel import register

# Phoenix 시작 및 트레이서 등록
px.launch_app()
tracer_provider = register()

# 이후 Ragas 평가가 자동으로 추적됨
```

---

## Amazon Bedrock 에이전트 평가

```python
from ragas.integrations.aws import convert_to_ragas_messages
from ragas.metrics import AspectCritic, RubricsScore

# 에이전트 아키텍처
# Knowledge Base (OpenSearch) + DynamoDB + Lambda + Agent

# 멀티턴 대화 변환
ragas_messages = convert_to_ragas_messages(bedrock_trace)

# 커스텀 메트릭 정의
accuracy_metric = AspectCritic(
    name="accuracy",
    definition="응답이 정확한 정보를 제공하는가?"
)

quality_metric = RubricsScore(
    name="quality",
    rubrics={...}
)
```

---

## Griptape 통합

```python
from ragas.integrations.griptape import transform_to_ragas_dataset

# 변환 함수로 Ragas 데이터셋 구성
ragas_dataset = transform_to_ragas_dataset(griptape_results)

# 검색 메트릭
retrieval_metrics = [context_precision, context_recall, context_relevance]

# 생성 메트릭
generation_metrics = [factual_correctness, faithfulness, response_groundedness]

# 평가 결과 예시
# context_precision: 1.0
# faithfulness: 1.0
```

---

## R2R 통합

```python
from r2r import R2RClient
from ragas.integrations.r2r import transform_to_ragas_dataset

# R2R 클라이언트 설정
client = R2RClient()

# RAG 엔드포인트 호출
results = client.rag(
    query="...",
    search_settings={"top_k": 5},
    rag_generation_config={"model": "openai/gpt-4o"}
)

# Ragas 데이터셋으로 변환
dataset = transform_to_ragas_dataset(results)

# 평가
result = evaluate(
    dataset,
    metrics=[response_relevancy, context_precision, faithfulness]
)

# 결과 업로드 (시각화)
result.upload()
```
