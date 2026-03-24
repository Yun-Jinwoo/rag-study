# LangChain 개요 (Overview)

> 공식 문서: https://docs.langchain.com/

---

## 1. LangChain이란?

LangChain은 **에이전트(Agent) 아키텍처와 다양한 모델·도구 통합 기능을 갖춘 오픈소스 프레임워크**다.
OpenAI, Anthropic, Google 등 주요 LLM 제공업체에 **10줄 이하의 코드**로 연결할 수 있으며,
빠른 에이전트 개발을 위한 표준화된 인터페이스를 제공한다.

---

## 2. LangChain 생태계 구성요소

| 프레임워크 | 설명 | 추천 대상 |
|---|---|---|
| **Deep Agents** | 컨텍스트 압축, 서브에이전트 스폰 등 고급 기능 내장 | 빠른 에이전트 개발 입문자 |
| **LangChain** | 커스터마이징 가능한 에이전트 빌딩 블록 제공 | 빠른 개발 + 적절한 커스터마이징 |
| **LangGraph** | 저수준 오케스트레이션, 복잡한 상태 관리 | 세밀한 워크플로우 제어가 필요한 개발자 |
| **LangSmith** | 트레이싱, 평가, 디버깅, 배포 플랫폼 | 프로덕션 운영 및 품질 관리 |

> LangChain 에이전트는 내부적으로 LangGraph 위에서 동작하며,
> 영속성(persistence), 스트리밍(streaming), 휴먼-인-더-루프(human-in-the-loop) 등을 지원한다.

---

## 3. 설치

```bash
pip install -U langchain

# 특정 LLM 제공업체 추가 설치
pip install -U langchain-openai
pip install -U langchain-anthropic
pip install -U langchain-google-genai

# 또는 uv 사용
uv add langchain
```

- **Python 3.10 이상** 필요

---

## 4. 최소 에이전트 예제 (Hello World)

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

---

## 5. 프로덕션 에이전트 6단계

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.memory import InMemorySaver
from pydantic import BaseModel

# 1. 시스템 프롬프트 정의
system_prompt = "You are a helpful assistant. Be concise and accurate."

# 2. 도구 정의
@tool
def search(query: str) -> str:
    """Search for information on the web."""
    return f"Search results for: {query}"

# 3. 모델 초기화
model = init_chat_model("claude-sonnet-4-6", temperature=0, max_tokens=1000)

# 4. 구조화된 출력 정의
class Response(BaseModel):
    answer: str
    confidence: float

# 5. 메모리(상태) 설정
memory = InMemorySaver()

# 6. 에이전트 조합 & 실행
agent = create_agent(
    model=model,
    tools=[search],
    system_prompt=system_prompt,
    checkpointer=memory,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about LangChain"}]},
    config={"configurable": {"thread_id": "session-001"}}
)
```

---

## 6. LangChain의 핵심 장점

1. **표준 모델 인터페이스**: 모든 LLM 제공업체에 통일된 API 제공
2. **유연한 에이전트 아키텍처**: 단순 시작점 + 광범위한 커스터마이징
3. **LangGraph 기반**: 영속적 실행 및 상태 관리 지원
4. **LangSmith 통합**: 디버깅·트레이싱 툴 내장
5. **풍부한 통합 생태계**: OpenAI, Anthropic, Google, AWS, HuggingFace 등

---

## 7. 주요 컴포넌트 목록

| 컴포넌트 | 역할 |
|---|---|
| Chat Models | LLM과의 표준화된 대화 인터페이스 |
| Prompt Templates | 재사용 가능한 프롬프트 구성 |
| Output Parsers | 모델 출력 파싱/구조화 |
| Document Loaders | 외부 데이터 소스 로딩 |
| Text Splitters | 문서 청킹 (chunking) |
| Embedding Models | 텍스트 → 벡터 변환 |
| Vector Stores | 벡터 저장·검색 |
| Retrievers | 관련 문서 검색 |
| Agents | LLM + 도구 + 루프 |
| Tools | 에이전트가 사용할 수 있는 함수 |
| Memory | 대화 상태 관리 |
| Chains | 컴포넌트들의 순차적 파이프라인 |

---

## 8. 관련 링크

- 공식 문서: https://docs.langchain.com/
- Python API 참조: https://reference.langchain.com/python/
- LangSmith: https://smith.langchain.com/
- GitHub: https://github.com/langchain-ai/langchain
