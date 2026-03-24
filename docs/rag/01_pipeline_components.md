# RAG 파이프라인 설계 및 구성요소

> 개인 학습 정리 노트

---

## 1. 전체 파이프라인 구조

RAG 파이프라인은 크게 **인덱싱(Indexing)** 단계와 **추론(Inference)** 단계로 나뉜다.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [인덱싱 단계] - 사전 작업 (오프라인)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
원본 문서
   ↓
Document Loader   (문서 로드)
   ↓
Text Splitter     (문서 청킹)
   ↓
Embedding Model   (벡터 변환)
   ↓
Vector Store      (벡터 저장)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [추론 단계] - 실시간 처리 (온라인)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용자 질문 (Query)
   ↓
Embedding Model   (질문 벡터화)
   ↓
Retriever         (관련 문서 검색)
   ↓
Prompt Template   (프롬프트 구성)
   ↓
LLM               (답변 생성)
   ↓
Output Parser     (출력 후처리)
   ↓
최종 답변
```

---

## 2. 컴포넌트별 상세 설명

### 2-1. Document Loader (문서 로더)

다양한 형식의 문서를 불러와 텍스트로 변환하는 역할.

| 포맷 | 예시 도구 |
|------|-----------|
| PDF | PyPDFLoader, PDFPlumberLoader |
| 웹페이지 | WebBaseLoader, SeleniumURLLoader |
| Word | Docx2txtLoader |
| CSV / Excel | CSVLoader |
| 데이터베이스 | SQLDatabaseLoader |
| 코드 | GitLoader |

```python
# LangChain 예시
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
docs = loader.load()  # Document 객체 리스트 반환
```

---

### 2-2. Text Splitter (텍스트 분할기)

로드된 문서를 **적절한 크기의 청크(chunk)** 로 나누는 역할.
LLM의 컨텍스트 윈도우 한계 + 검색 정밀도 향상을 위해 필수.

**주요 파라미터:**
- `chunk_size` : 청크 최대 길이 (토큰 또는 문자 수)
- `chunk_overlap` : 청크 간 겹치는 부분 (문맥 손실 방지)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

**주요 분할 전략:**

| 방법 | 설명 |
|------|------|
| RecursiveCharacterTextSplitter | `\n\n` → `\n` → ` ` 순서로 재귀적 분할 (범용) |
| CharacterTextSplitter | 특정 구분자 기준 단순 분할 |
| TokenTextSplitter | 토큰 기준 분할 (LLM 토큰 한계 정확히 지킴) |
| MarkdownHeaderTextSplitter | 마크다운 헤더 기준 분할 |
| SemanticChunker | 의미 유사도 기반 분할 (고급) |

**청킹 전략 선택 팁:**
- 일반 텍스트 → `RecursiveCharacterTextSplitter`
- 코드 → `Language` 기반 splitter
- 구조화된 문서(논문, 보고서) → `MarkdownHeaderTextSplitter`
- 정밀도 중요 → `SemanticChunker`

---

### 2-3. Embedding Model (임베딩 모델)

텍스트를 **고차원 숫자 벡터**로 변환하는 모델.

**주요 임베딩 모델:**

| 모델 | 제공사 | 특징 |
|------|--------|------|
| text-embedding-3-small | OpenAI | 가성비 좋음, 1536차원 |
| text-embedding-3-large | OpenAI | 고성능, 3072차원 |
| embed-english-v3.0 | Cohere | 영어 특화 |
| multilingual-e5-large | HuggingFace | 다국어, 무료 |
| ko-sroberta-multitask | HuggingFace | 한국어 특화 |
| bge-m3 | BAAI | 다국어, 오픈소스 고성능 |

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("RAG란 무엇인가?")
# → [0.021, -0.013, 0.087, ...]  (1536차원 벡터)
```

> **한국어 사용 시 주의:** 한국어 지원이 좋은 임베딩 모델을 선택해야 검색 품질이 올라간다.

---

### 2-4. Vector Store (벡터 저장소)

임베딩된 벡터를 저장하고 유사도 검색을 수행하는 데이터베이스.

**주요 Vector DB 비교:**

| DB | 특징 | 사용 환경 |
|----|------|-----------|
| **FAISS** | Meta 제작, 로컬, 빠름, 무료 | 소규모 / 실험용 |
| **Chroma** | 로컬/서버, 사용 쉬움, 오픈소스 | 개발/프로토타입 |
| **Pinecone** | 완전 관리형 클라우드, 확장성 | 프로덕션 |
| **Weaviate** | 오픈소스, 클라우드/셀프호스팅 | 중대형 서비스 |
| **Qdrant** | 오픈소스, 고성능, Rust 기반 | 중대형 서비스 |
| **Milvus** | 대규모 엔터프라이즈용 | 대규모 서비스 |

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)
```

---

### 2-5. Retriever (검색기)

Vector Store에서 질문과 관련된 문서를 찾아오는 역할.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",  # "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4}     # 상위 4개 문서 반환
)

results = retriever.invoke("RAG의 핵심 개념은?")
```

**검색 방식 종류:**

| 방식 | 설명 |
|------|------|
| Similarity Search | 코사인 유사도 기준 top-K |
| MMR (Maximal Marginal Relevance) | 유사도 + 다양성 균형 |
| Similarity Score Threshold | 유사도 임계값 이상만 반환 |
| BM25 (키워드 검색) | 전통적 역인덱스 기반 |
| **Hybrid Search** | 벡터 + 키워드 검색 결합 (권장) |

---

### 2-6. Prompt Template (프롬프트 템플릿)

검색된 문서와 사용자 질문을 조합해 LLM에 전달할 프롬프트를 구성.

```python
from langchain_core.prompts import ChatPromptTemplate

template = """당신은 주어진 문서를 기반으로 질문에 답변하는 AI입니다.
아래 문서를 참고하여 질문에 답하세요. 문서에 없는 내용은 모른다고 하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]"""

prompt = ChatPromptTemplate.from_template(template)
```

---

### 2-7. LLM (언어 모델)

프롬프트를 받아 최종 답변을 생성하는 핵심 모델.

**주요 LLM:**

| 모델 | 제공사 | 특징 |
|------|--------|------|
| GPT-4o, GPT-4o-mini | OpenAI | 범용 고성능 |
| Claude 3.5 Sonnet | Anthropic | 긴 문서, 코드 강점 |
| Gemini 1.5 Pro | Google | 긴 컨텍스트 강점 |
| Llama 3 | Meta | 오픈소스, 로컬 실행 가능 |
| EXAONE, HyperCLOVA | LG, Naver | 한국어 특화 |

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

---

### 2-8. Output Parser (출력 파서)

LLM의 출력을 원하는 형식으로 변환.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()  # 문자열로 추출
```

JSON, Pydantic 모델 등 구조화된 출력도 가능.

---

## 3. 전체 체인 조합 (LangChain LCEL)

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("RAG가 무엇인가요?")
print(answer)
```

---

## 4. 파이프라인 설계 시 고려사항

### 청킹 전략
- chunk_size가 너무 크면 → 노이즈 증가, 검색 정밀도 하락
- chunk_size가 너무 작으면 → 문맥 손실, 의미 단절
- **일반적으로 500~1000 토큰, overlap 10~20% 권장**

### 검색 품질 향상
- 키워드 검색(BM25) + 벡터 검색을 혼합하는 **Hybrid Search** 적용
- **Re-ranker** 모델로 검색 결과 재정렬 (Cohere Reranker 등)
- Top-K 값을 너무 크게 설정하면 LLM 컨텍스트 초과 주의

### 프롬프트 설계
- "문서에 없는 내용은 모른다고 하세요" 같은 **Grounding 지시어** 필수
- 언어(한국어/영어) 명시
- 답변 형식(번호 목록, 표 등) 지정 가능

### 평가 (Ragas)
- 파이프라인 구현 후 **Faithfulness, Answer Relevancy, Context Recall** 등으로 품질 측정
- 평가 결과를 바탕으로 청킹/검색/프롬프트 튜닝

---

## 5. 기술 스택 선택 가이드 (학습용 추천)

| 컴포넌트 | 학습용 추천 |
|----------|------------|
| Document Loader | LangChain Community Loaders |
| Text Splitter | RecursiveCharacterTextSplitter |
| Embedding | OpenAI text-embedding-3-small 또는 HuggingFace bge-m3 |
| Vector Store | Chroma (로컬, 가장 간단) |
| LLM | GPT-4o-mini (비용 저렴) 또는 Llama3 (무료) |
| 프레임워크 | LangChain + LangGraph |
| 평가 | Ragas |

---

## 참고 자료

- [LangChain RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/)
- [Ragas 공식 문서](https://docs.ragas.io/en/stable/)
- [Advanced RAG 기법 정리 (블로그)](https://blog.langchain.dev/deconstructing-rag/)
