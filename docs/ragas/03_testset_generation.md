# Ragas 테스트셋 생성 (Testset Generation)

---

## 개요

Ragas는 **합성 테스트 데이터셋**을 자동으로 생성하는 도구를 제공한다.
수작업 데이터셋 구축의 한계를 극복하고 다양한 시나리오를 체계적으로 커버할 수 있다.

---

## 테스트셋 생성 아키텍처

```
원본 문서 (Documents)
        ↓
[Knowledge Graph 구축]
  - 문서 분석
  - 청크 생성
  - 노드/관계 구성
        ↓
[Transforms 적용]
  - 요약 추출
  - 핵심 구문 추출
  - 임베딩 생성
  - 유사도 관계 구축
        ↓
[Synthesizers로 질문 생성]
  - SingleHop 질문
  - MultiHop 질문
        ↓
합성 테스트셋 (EvaluationDataset)
```

---

## Knowledge Graph (지식 그래프)

### 노드 타입 (NodeType)

```python
from ragas.testset.graph import NodeType

NodeType.UNKNOWN   # 알 수 없는 타입
NodeType.DOCUMENT  # 문서 수준 노드
NodeType.CHUNK     # 청크 수준 노드
```

### 주요 클래스

```python
from ragas.testset.graph import Node, Relationship, KnowledgeGraph

# 노드 생성
node = Node(
    id="unique_id",
    type=NodeType.CHUNK,
    properties={
        "content": "청크 내용...",
        "metadata": {"source": "document.pdf"}
    }
)

# 관계 생성
relation = Relationship(
    source=node1,
    target=node2,
    type="similar_to",
    properties={"score": 0.85}
)

# 지식 그래프 생성 및 관리
kg = KnowledgeGraph()
kg.add(node)
kg.save("knowledge_graph.json")      # 저장
kg = KnowledgeGraph.load("kg.json")  # 로드
```

### KnowledgeGraph 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `add()` | 노드 또는 관계 추가 |
| `save(path)` | 파일로 저장 |
| `load(path)` | 파일에서 로드 |
| `get_node_by_id(id)` | ID로 노드 검색 |
| `find_indirect_clusters()` | 간접 연결된 클러스터 탐색 |
| `remove_node(node)` | 노드 제거 |
| `find_two_nodes_single_rel()` | 단일 관계로 연결된 두 노드 탐색 |

---

## Transforms (데이터 변환)

### Extractors (추출기)

```python
from ragas.testset.transforms import (
    SummaryExtractor,       # 문서 요약 추출
    KeyphrasesExtractor,    # 핵심 구문 추출
    TitleExtractor,         # 제목 추출
    HeadlinesExtractor,     # 헤드라인 추출
    EmbeddingExtractor,     # 임베딩 생성
)
```

### Relationship Builders (관계 구축기)

```python
from ragas.testset.transforms import (
    CosineSimilarityBuilder,  # 코사인 유사도 기반 관계
    JaccardSimilarityBuilder, # 자카드 유사도 기반 관계
)
```

### Transform 적용

```python
from ragas.testset.transforms import apply_transforms, Parallel

# 기본 transforms 사용
from ragas.testset.transforms import default_transforms

transforms = default_transforms(
    documents=documents,
    llm=llm,
    embedding_model=embedding_model
)

# 사전 청킹된 데이터에 최적화된 transforms
from ragas.testset.transforms import default_transforms_for_prechunked

transforms = default_transforms_for_prechunked(
    documents=documents,
    llm=llm,
    embedding_model=embedding_model
)

# 병렬 처리
parallel_transforms = Parallel(transform1, transform2)

# 적용
await apply_transforms(kg, transforms)
```

---

## Synthesizers (질문 합성기)

### 질문 유형

```python
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,   # 단일 문서 기반 구체적 질문
    MultiHopAbstractQuerySynthesizer,    # 다중 문서 기반 추상적 질문
    MultiHopSpecificQuerySynthesizer,    # 다중 문서 기반 구체적 질문
)
```

| Synthesizer | 질문 특성 | 난이도 |
|-------------|-----------|--------|
| SingleHop Specific | 하나의 청크에서 답변 가능 | 쉬움 |
| MultiHop Abstract | 여러 청크를 종합해 추론 필요 | 어려움 |
| MultiHop Specific | 여러 청크의 구체적 정보 결합 | 중간 |

### 기본 질문 분포

```python
from ragas.testset.synthesizers import default_query_distribution

# 균형잡힌 질문 분포 자동 생성
query_distribution = default_query_distribution(llm)
```

---

## 테스트셋 생성 전체 예시

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LLM 및 임베딩 설정
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# 테스트셋 생성기 초기화
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

# 테스트셋 생성 (100개 샘플)
testset = generator.generate_with_langchain_docs(
    documents=documents,  # LangChain Document 객체 목록
    testset_size=100,
)

# EvaluationDataset으로 변환
eval_dataset = testset.to_evaluation_dataset()
```

---

## 커스터마이징 옵션

### 1. 비영어 테스트셋 생성

```python
# 한국어 테스트셋 생성
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

synthesizer = SingleHopSpecificQuerySynthesizer(
    llm=llm,
    language="korean"
)
```

### 2. 페르소나(Persona) 기반 생성

```python
from ragas.testset.persona import Persona

personas = [
    Persona(
        name="초보 학습자",
        role_description="RAG에 대해 처음 배우는 학생"
    ),
    Persona(
        name="전문 개발자",
        role_description="RAG 시스템을 구현하는 시니어 엔지니어"
    )
]

# 페르소나를 고려한 질문 생성
generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=50,
    query_distribution=query_distribution,
    personas=personas
)
```

### 3. 사전 청킹된 데이터 처리

```python
# 이미 청킹된 문서가 있는 경우
from ragas.testset.transforms import default_transforms_for_prechunked

transforms = default_transforms_for_prechunked(
    documents=chunked_documents,
    llm=llm,
    embedding_model=embedding_model
)
```

---

## 이상적인 테스트셋 구성 요소

```
좋은 테스트셋
├── 사실 기반 질문 (Factual)
│   "Einstein은 언제 태어났나?"
│
├── 추론 필요 질문 (Reasoning)
│   "A와 B의 차이점은 무엇인가?"
│
├── 다중 문서 질문 (Multi-hop)
│   "X의 발명이 Y 분야에 미친 영향은?"
│
├── 경계 케이스 (Edge Cases)
│   문서에 없는 정보에 대한 질문
│
└── 모호한 질문 (Ambiguous)
    여러 해석이 가능한 질문
```
