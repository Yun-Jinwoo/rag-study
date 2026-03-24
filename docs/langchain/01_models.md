# LangChain 모델 (Models / Chat Models)

> 공식 문서: https://docs.langchain.com/oss/python/langchain/models

---

## 1. 개요

LangChain의 모델은 **에이전트의 추론 엔진**으로, 어떤 도구를 언제 사용할지 결정하고 최종 응답을 생성한다.
LangChain은 OpenAI, Anthropic, Google, Azure, AWS Bedrock 등 **모든 주요 LLM 제공업체**에 표준화된 인터페이스를 제공한다.

### 주요 기능
- **Tool Calling**: 외부 도구(DB, API 등) 호출 가능
- **Structured Output**: 정의된 형식으로 응답 강제
- **Multimodality**: 텍스트 외 이미지, 오디오, 비디오 처리
- **Reasoning**: 다단계 추론으로 결론 도출

---

## 2. 모델 초기화

### `init_chat_model()` 사용 (권장)

```python
from langchain.chat_models import init_chat_model

# OpenAI
model = init_chat_model("gpt-4o")

# Anthropic
model = init_chat_model("claude-sonnet-4-6")

# 파라미터 지정
model = init_chat_model(
    "gpt-4o",
    temperature=0.2,
    max_tokens=2000,
    timeout=30,
    max_retries=3
)
```

### 제공업체별 패키지 설치

```bash
pip install langchain-openai        # OpenAI
pip install langchain-anthropic     # Anthropic
pip install langchain-google-genai  # Google Gemini
pip install langchain-aws           # AWS Bedrock
pip install langchain-ollama        # Ollama (로컬 모델)
```

---

## 3. 주요 파라미터

| 파라미터 | 설명 | 예시 |
|---|---|---|
| `model` | 모델 ID | `"gpt-4o"`, `"claude-sonnet-4-6"` |
| `temperature` | 창의성 제어 (0=결정적, 1=창의적) | `0.0` ~ `1.0` |
| `max_tokens` | 최대 출력 토큰 수 | `2000` |
| `timeout` | 응답 최대 대기 시간 (초) | `30` |
| `max_retries` | 실패 시 재시도 횟수 | `6` (기본값) |
| `api_key` | 인증 키 | `"sk-..."` |

---

## 4. 호출 방법

### 4-1. `invoke()` — 단일 응답

```python
# 단순 텍스트
response = model.invoke("Why do parrots have colorful feathers?")
print(response.content)

# 대화 히스토리 포함
from langchain.schema import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is RAG?"),
    AIMessage(content="RAG stands for Retrieval-Augmented Generation..."),
    HumanMessage(content="Can you give an example?"),
]
response = model.invoke(messages)
```

### 4-2. `stream()` — 실시간 스트리밍

```python
for chunk in model.stream("Explain quantum computing"):
    print(chunk.content, end="|", flush=True)
# 반환 타입: AIMessageChunk (누적 합산 가능)
```

### 4-3. `batch()` — 일괄 처리

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?"
])

# 완료되는 순서대로 받기
for response in model.batch_as_completed(["Q1", "Q2", "Q3"]):
    print(response)

# 병렬 처리 수 제한
responses = model.batch(questions, config={"max_concurrency": 5})
```

---

## 5. Tool Calling (도구 호출)

모델이 외부 도구를 호출하는 흐름:
1. 사용자 입력 제공
2. 모델이 어떤 도구를 호출할지 결정
3. 도구 실행 (병렬 가능)
4. 결과를 모델에 전달 → 최종 응답 생성

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

# 모델에 도구 바인딩
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Boston?")

# tool_calls 확인
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': '...'}]

# 강제로 도구 사용하게 하기
model_with_tools = model.bind_tools([get_weather], tool_choice="any")

# 특정 도구 강제 지정
model_with_tools = model.bind_tools([get_weather], tool_choice="get_weather")

# 병렬 도구 호출 비활성화
model_with_tools = model.bind_tools([t1, t2], parallel_tool_calls=False)
```

---

## 6. Structured Output (구조화된 출력)

### Pydantic 모델 사용 (권장)

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    director: str
    rating: float

model_with_structure = model.with_structured_output(Movie)
result = model_with_structure.invoke("Tell me about Inception")
# result: Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8)
```

### TypedDict 사용

```python
from typing_extensions import TypedDict

class MovieDict(TypedDict):
    title: str
    year: int
    director: str

model_with_structure = model.with_structured_output(MovieDict)
```

### JSON Schema 사용

```python
json_schema = {
    "title": "Movie",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["title", "year"]
}
model_with_structure = model.with_structured_output(json_schema)
```

---

## 7. 고급 기능

### 모델 프로파일 (capability 확인)

```python
print(model.profile)
# {
#   "max_input_tokens": 128000,
#   "image_inputs": True,
#   "tool_calling": True,
#   "structured_output": True,
# }
```

### Runtime 설정

```python
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",
        "tags": ["humor"],
        "max_concurrency": 5
    }
)
```

### 토큰 사용량 추적

```python
response = model.invoke("Hello")
print(response.response_metadata["usage"])
# {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60}
```

### Rate Limiting

```python
from langchain.callbacks import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=1)
model = init_chat_model("gpt-4o", rate_limiter=rate_limiter)
```

### Prompt Caching (비용 절감)

Anthropic, OpenAI, Gemini, AWS Bedrock에서 지원.
반복되는 긴 프롬프트(예: 시스템 프롬프트, 긴 컨텍스트)에 자동 또는 명시적 캐싱 적용 가능.

---

## 8. 지원 제공업체

| 제공업체 | 패키지 | 대표 모델 |
|---|---|---|
| OpenAI | `langchain-openai` | `gpt-4o`, `gpt-4.1` |
| Anthropic | `langchain-anthropic` | `claude-sonnet-4-6`, `claude-opus-4-6` |
| Google | `langchain-google-genai` | `gemini-2.0-flash` |
| Azure OpenAI | `langchain-openai` | Azure 배포 모델 |
| AWS Bedrock | `langchain-aws` | Bedrock 지원 모델 |
| Ollama (로컬) | `langchain-ollama` | Llama, Mistral 등 |
| HuggingFace | `langchain-huggingface` | 오픈소스 모델 |
| Groq | `langchain-groq` | LLaMA, Mixtral |
