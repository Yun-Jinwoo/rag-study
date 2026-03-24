# LangGraph — 저수준 에이전트 오케스트레이션

> 공식 문서: https://docs.langchain.com/oss/python/langgraph/overview

---

## 1. LangGraph란?

LangGraph는 **장기 실행(long-running), 상태 유지(stateful) 에이전트를 구축·관리·배포하기 위한 저수준 오케스트레이션 프레임워크**다.

### LangChain vs LangGraph

| 항목 | LangChain | LangGraph |
|---|---|---|
| 추상화 수준 | 높음 (빠른 시작) | 낮음 (세밀한 제어) |
| 학습 난이도 | 쉬움 | 보통 |
| 커스터마이징 | 제한적 | 무제한 |
| 프롬프트 추상화 | 제공 | 없음 |
| 아키텍처 추상화 | 제공 | 없음 |
| 적합한 경우 | 빠른 에이전트 개발 | 복잡한 결정론적 워크플로우 |

> LangChain 에이전트는 내부적으로 LangGraph 위에서 동작한다.
> LangGraph를 사용하기 위해 LangChain이 반드시 필요하지는 않다.

---

## 2. 핵심 개념

### 2-1. StateGraph (상태 그래프)

워크플로우를 **노드(Node)**와 **엣지(Edge)**로 구성하는 그래프.

```python
from langgraph.graph import StateGraph, MessagesState, START, END

# StateGraph 생성
graph_builder = StateGraph(MessagesState)
```

### 2-2. State (상태)

그래프가 실행되는 동안 공유하는 데이터 구조.

```python
from typing import TypedDict, Annotated
from langchain.schema import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_id: str
    iteration_count: int
```

`MessagesState`는 기본 제공되는 상태 스키마로 `messages` 필드를 포함한다.

### 2-3. Nodes (노드)

실제 작업을 수행하는 함수. 상태를 입력으로 받아 업데이트된 상태를 반환.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

def call_model(state: AgentState):
    """LLM 호출 노드."""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def process_tool_calls(state: AgentState):
    """도구 호출 처리 노드."""
    last_message = state["messages"][-1]
    # 도구 실행 로직
    return {"messages": [tool_result]}
```

### 2-4. Edges (엣지)

노드 간의 연결. 조건부 엣지로 동적 라우팅 가능.

```python
from langgraph.graph import END

# 일반 엣지
graph_builder.add_edge(START, "call_model")
graph_builder.add_edge("call_model", END)

# 조건부 엣지
def should_continue(state: AgentState) -> str:
    """도구 호출이 있으면 도구 노드로, 없으면 종료."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "call_model",
    should_continue,
    {"tools": "tool_node", END: END}
)
```

### 2-5. Compilation & Execution

```python
# 그래프 컴파일
graph = graph_builder.compile()

# 실행
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

---

## 3. 전체 예제 — ReAct 에이전트

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# 1. 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's sunny in {city}!"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tools = [get_weather, calculate]

# 2. 모델 + 도구 바인딩
model = init_chat_model("gpt-4o")
model_with_tools = model.bind_tools(tools)

# 3. 노드 함수 정의
def agent_node(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 4. 그래프 구성
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()

# 5. 실행
result = graph.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seoul?"}]
})
print(result["messages"][-1].content)
```

---

## 4. 주요 기능

### 4-1. Persistence (영속성)

체크포인터(Checkpointer)를 사용해 상태를 저장하고 중단된 지점에서 재개.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# 인메모리 (개발용)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# PostgreSQL (프로덕션)
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph = graph_builder.compile(checkpointer=checkpointer)

# thread_id로 대화 이어가기
config = {"configurable": {"thread_id": "user-123-session-001"}}
result = graph.invoke({"messages": [...]}, config=config)
```

### 4-2. Human-in-the-Loop (인간 개입)

실행 중 특정 지점에서 사람의 승인이나 수정을 요청.

```python
from langgraph.types import interrupt

def risky_action_node(state):
    """중요한 작업 전 승인 요청."""
    action = state["planned_action"]

    # 실행 중단 및 사람에게 확인 요청
    approval = interrupt({
        "message": f"다음 작업을 수행할까요? {action}",
        "action": action
    })

    if approval["approved"]:
        # 실제 작업 수행
        return {"result": execute_action(action)}
    else:
        return {"result": "사용자가 취소했습니다."}
```

### 4-3. Streaming (스트리밍)

```python
# 상태값 스트리밍
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Research AI trends"}]},
    stream_mode="values"
):
    print(chunk["messages"][-1])

# 이벤트 스트리밍 (더 세밀한 제어)
async for event in graph.astream_events(input_data, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

### 4-4. Time-Travel Debugging (시간 여행 디버깅)

과거 상태로 돌아가서 다른 경로 실험 가능.

```python
# 이전 상태 목록 조회
history = list(graph.get_state_history(config))

# 특정 시점으로 복원
past_state = history[2]
graph.update_state(config, past_state.values, as_node="agent")

# 해당 시점부터 재실행
result = graph.invoke(None, config=config)
```

---

## 5. 설치

```bash
pip install -U langgraph

# 체크포인터 추가 (프로덕션)
pip install langgraph-checkpoint-postgres  # PostgreSQL
pip install langgraph-checkpoint-sqlite    # SQLite
```

---

## 6. LangSmith 통합 (트레이싱)

```bash
export LANGSMITH_API_KEY="your-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="my-rag-project"
```

설정 후 모든 그래프 실행이 자동으로 LangSmith에 트레이싱된다.

---

## 7. 언제 LangGraph를 쓸까?

**LangChain (create_agent) 사용 권장:**
- 빠른 프로토타이핑
- 표준적인 ReAct 패턴
- 커스터마이징 요구사항이 많지 않을 때

**LangGraph 사용 권장:**
- 복잡한 다단계 워크플로우
- 결정론적 분기 로직이 많을 때
- 세밀한 상태 관리 필요
- 휴먼-인-더-루프 구현
- 병렬 실행 제어
- 시간 여행 디버깅 필요
