# LangChain 에이전트 & 도구 (Agents & Tools)

> 공식 문서:
> - Agents: https://docs.langchain.com/oss/python/langchain/agents
> - Tools: https://docs.langchain.com/oss/python/langchain/tools

---

## 1. 에이전트(Agent)란?

에이전트는 **LLM + 도구(Tools) + 루프(Loop)**의 조합이다.
목표를 달성하기 위해 도구를 반복 호출하며, 종료 조건(최종 응답 생성 or 최대 반복)에 도달하면 멈춘다.

### ReAct 패턴 (Reasoning + Acting)

```
사용자 입력
    ↓
[Reasoning] 어떤 도구를 써야 할까?
    ↓
[Acting] 도구 호출
    ↓
[Observation] 결과 확인
    ↓
[Reasoning] 충분한가? 더 필요한가?
    ↓ (필요하면 반복)
최종 응답 생성
```

### 실제 예시 (무선 헤드폰 추천)

```
Q: "가장 인기 있는 무선 헤드폰을 찾아서 재고를 확인해줘"

1. [Reasoning] 인기 순위는 시간에 따라 변하므로 검색 필요
2. [Acting]    search_products("wireless headphones")
3. [Observation] WH-1000XM5가 1위
4. [Reasoning] 재고 확인 필요
5. [Acting]    check_inventory("WH-1000XM5")
6. [Observation] 재고 10개 있음
7. [Final Answer] "WH-1000XM5가 가장 인기 있으며 현재 10개 재고 있습니다."
```

---

## 2. 에이전트 생성

### 기본 생성

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant.",
)
```

### 고급 설정

```python
from langchain.chat_models import init_chat_model
from langchain.memory import InMemorySaver

model = init_chat_model("gpt-4o", temperature=0.1, max_tokens=2000)
memory = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant.",
    name="my_agent",  # snake_case 권장
    checkpointer=memory,
)
```

### 에이전트 실행

```python
# 단일 실행
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Seoul?"}]}
)
print(result["messages"][-1].content)

# 대화 컨텍스트 유지 (thread_id 사용)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "And what about Tokyo?"}]},
    config={"configurable": {"thread_id": "conv-001"}}
)

# 스트리밍
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news"}]},
    stream_mode="values"
):
    latest = chunk["messages"][-1]
    if hasattr(latest, "tool_calls") and latest.tool_calls:
        print(f"호출 중인 도구: {[tc['name'] for tc in latest.tool_calls]}")
    else:
        print(f"응답: {latest.content}")
```

---

## 3. 도구(Tools) 정의

### 3-1. `@tool` 데코레이터 (기본)

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: The search query string
        limit: Maximum number of results to return
    """
    # 실제 구현
    return f"Found {limit} results for '{query}'"
```

> **중요:** 타입 힌트는 **필수**다. 모델이 인자 스키마를 이해하는 데 사용됨.
> 함수 docstring이 자동으로 도구 설명이 된다.

### 3-2. 커스텀 이름/설명

```python
@tool("web_search", description="Search the internet for current information.")
def search(query: str) -> str:
    """Perform a web search."""
    return f"Results: ..."
```

### 3-3. 복잡한 인자 — Pydantic 스키마

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    language: str = Field(default="en", description="Language code")
    max_results: int = Field(default=5, ge=1, le=20)

@tool(args_schema=SearchParams)
def advanced_search(query: str, language: str = "en", max_results: int = 5) -> str:
    """Advanced search with filtering options."""
    return f"Searching '{query}' in {language}..."
```

---

## 4. 도구 런타임 컨텍스트 (ToolRuntime)

도구 내부에서 에이전트 상태, 사용자 컨텍스트, 영속 저장소에 접근 가능하다.

```python
from langchain.tools import tool, ToolRuntime

@tool
def personalized_search(query: str, runtime: ToolRuntime) -> str:
    """Search with user-specific context."""
    # 대화 상태 접근
    messages = runtime.state["messages"]

    # 사용자 컨텍스트 접근 (invoke 시 전달된 값)
    user_id = runtime.context.get("user_id", "anonymous")

    # 장기 기억 저장소
    store = runtime.store
    user_prefs = store.get(("users", user_id), "preferences")

    return f"Personalized results for user {user_id}: ..."

# invoke 시 컨텍스트 전달
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for Python tutorials"}]},
    config={"configurable": {"context": {"user_id": "user-123"}}}
)
```

### 스트리밍 업데이트 (진행 상황 실시간 전달)

```python
@tool
def long_running_task(data: str, runtime: ToolRuntime) -> str:
    """Process data with progress updates."""
    runtime.stream_writer({"status": "시작 중..."})
    # 처리 로직
    runtime.stream_writer({"status": "50% 완료"})
    return "처리 완료"
```

### 상태 업데이트 (Command 반환)

```python
from langchain.agents import Command

@tool
def save_preference(preference: str, runtime: ToolRuntime):
    """Save user preference to state."""
    return Command(
        update={"user_preferences": preference},
        content=f"Preference saved: {preference}"
    )
```

---

## 5. 도구 에러 핸들링

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.schema import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch and handle tool execution errors gracefully."""
    try:
        return handler(request)
    except ValueError as e:
        return ToolMessage(
            content=f"입력 오류: {str(e)}. 올바른 형식으로 다시 시도해주세요.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        return ToolMessage(
            content=f"도구 실행 실패: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(model, tools, middleware=[handle_tool_errors])
```

---

## 6. 동적 모델 선택

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def dynamic_model_selection(request, handler):
    """Use cheaper model for simple tasks, expensive model for complex ones."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 긴 대화 → 더 강력한 모델
        return handler(request.override(model=advanced_model))
    else:
        return handler(request.override(model=basic_model))
```

---

## 7. 동적 프롬프트

```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def user_role_prompt(request) -> str:
    """Adapt system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "beginner")

    if user_role == "expert":
        return "Provide detailed technical responses with code examples."
    elif user_role == "manager":
        return "Provide high-level summaries without technical details."
    else:
        return "Explain concepts simply with analogies."
```

---

## 8. 구조화된 출력

### ToolStrategy (모든 도구 호출 모델 지원)

```python
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    response_format=ToolStrategy(ContactInfo)
)
```

### ProviderStrategy (네이티브 구조화 출력)

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

---

## 9. 내장 도구 (Prebuilt Tools)

LangChain은 즉시 사용 가능한 도구들을 제공한다:

```python
from langchain_community.tools import (
    TavilySearchResults,      # 웹 검색
    WikipediaQueryRun,        # 위키피디아 검색
    PythonREPLTool,           # Python 코드 실행
    SQLDatabaseTool,          # SQL 쿼리
    ArxivQueryRun,            # 논문 검색
)

search_tool = TavilySearchResults(max_results=5)
agent = create_agent(model, tools=[search_tool])
```

---

## 10. 네이밍 규칙

- 에이전트 이름: `snake_case` (`research_assistant`, `qa_bot`)
- 도구 이름: `snake_case` (`get_weather`, `search_database`)
- 공백, 특수문자 사용 금지 (제공업체 호환성 문제)
