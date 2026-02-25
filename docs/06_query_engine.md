# Step 6: Query Engine & LangChain Orchestration

## Overview

The Query Engine orchestrates the end-to-end RAG pipeline: query processing, retrieval, context assembly, LLM generation, and response formatting.

## Architecture

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Query Validator     │ (Length, sanitization)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Hybrid Retrieval    │ (Dense + BM25)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Re-ranker           │ (Cross-encoder)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Context Builder     │ (Format with sources)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Prompt Template     │ (System + context + query)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  LLM (GPT-4)         │ (Generate answer)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Response Formatter  │ (Answer + citations)
└──────────────────────┘
```

## LangChain Integration

### LCEL (LangChain Expression Language)

**Modern approach**: Compose chains using pipe operator

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# Define components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful documentation assistant."),
    ("human", "{query}")
])

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

output_parser = StrOutputParser()

# Compose chain
chain = prompt | llm | output_parser

# Invoke
response = chain.invoke({"query": "How do I...?"})
```

### RAG Chain with Retrieval

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

# Custom retriever
def retrieve_context(query: str) -> str:
    results = hybrid_search.search(query, top_k=5)
    results = reranker.rerank(query, results, top_n=3)
    return format_context(results)

# RAG chain
rag_chain = (
    RunnableParallel({
        "context": lambda x: retrieve_context(x["query"]),
        "query": lambda x: x["query"]
    })
    | prompt
    | llm
    | output_parser
)

response = rag_chain.invoke({"query": "..."})
```

---

## Prompt Engineering

### System Prompt

```python
SYSTEM_PROMPT = """You are an expert technical documentation assistant.

Your role is to answer questions based ONLY on the provided context from the documentation.

Guidelines:
1. Answer accurately based on the context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Cite sources using [Source: {source_name}] format
4. Be concise but complete
5. Use code examples from the context when relevant
6. If multiple solutions exist, present them with pros/cons

Never make up information not present in the context.
"""
```

### RAG Prompt Template

```python
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context from documentation:

{context}

---

Question: {query}

Answer:""")
])
```

### Few-Shot Examples (Optional)

```python
FEW_SHOT_EXAMPLES = """
Example 1:
Question: How do I install the CLI?
Context: The CLI can be installed via npm using 'npm install -g our-cli'.
Answer: You can install the CLI using npm with the command `npm install -g our-cli`. [Source: Installation Guide]

Example 2:
Question: What is the rate limit?
Context: The API has a rate limit of 100 requests per minute per user.
Answer: The API enforces a rate limit of 100 requests per minute per user. [Source: API Reference]
"""
```

---

## Context Window Management

### Token Budget Allocation

**GPT-4 Turbo Context Window**: 128,000 tokens

**Allocation**:
- System Prompt: ~300 tokens
- User Query: ~100 tokens
- Retrieved Context: ~3,000 tokens (3 chunks × 512 tokens × 2 for safety)
- Response Buffer: ~1,000 tokens
- **Total Used**: ~4,400 tokens (leaves plenty of headroom)

### Context Truncation

```python
import tiktoken

def truncate_context(context: str, max_tokens: int = 3000) -> str:
    """Truncate context to fit within token budget."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(context)

    if len(tokens) <= max_tokens:
        return context

    # Truncate and add indicator
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text + "\n\n[... context truncated ...]"
```

---

## Query Validation & Sanitization

### Input Validation

```python
class QueryValidator:
    """Validate and sanitize user queries."""

    MIN_LENGTH = 3
    MAX_LENGTH = 500

    @classmethod
    def validate(cls, query: str) -> tuple[bool, str]:
        """
        Validate query.

        Returns:
            (is_valid, error_message)
        """
        # Strip whitespace
        query = query.strip()

        # Length checks
        if len(query) < cls.MIN_LENGTH:
            return False, f"Query too short (min {cls.MIN_LENGTH} characters)"

        if len(query) > cls.MAX_LENGTH:
            return False, f"Query too long (max {cls.MAX_LENGTH} characters)"

        # Check for empty query
        if not query:
            return False, "Query cannot be empty"

        return True, ""

    @classmethod
    def sanitize(cls, query: str) -> str:
        """Sanitize query to prevent prompt injection."""
        # Remove control characters
        query = ''.join(char for char in query if ord(char) >= 32 or char == '\n')

        # Limit newlines
        while '\n\n\n' in query:
            query = query.replace('\n\n\n', '\n\n')

        return query.strip()
```

---

## Response Formatting

### Citation Extraction

```python
def format_response_with_citations(answer: str, sources: list) -> dict:
    """
    Format response with source citations.

    Args:
        answer: LLM-generated answer
        sources: List of source documents

    Returns:
        Formatted response dictionary
    """
    return {
        "answer": answer,
        "sources": [
            {
                "source": src['source'],
                "chunk_index": src['chunk_index'],
                "relevance_score": src.get('rerank_score', src.get('score', 0)),
                "text_preview": src['text'][:200] + "..."
            }
            for src in sources
        ],
        "metadata": {
            "num_sources": len(sources),
            "retrieval_method": "hybrid",
            "model": "gpt-4-turbo-preview"
        }
    }
```

---

## Fallback Strategies

### Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    """Circuit breaker for API failures."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result

        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        """Reset on successful call."""
        self.failures = 0
        self.state = "CLOSED"

    def on_failure(self):
        """Handle failure."""
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
```

### Graceful Degradation

```python
def query_with_fallback(query: str) -> dict:
    """Query with multiple fallback levels."""
    try:
        # Level 1: Full RAG pipeline
        return full_rag_query(query)

    except OpenAIAPIError as e:
        logger.warning(f"OpenAI API error, falling back to cached response: {e}")

        try:
            # Level 2: Semantic cache
            return get_cached_response(query)

        except CacheMissError:
            logger.warning("Cache miss, returning retrieval only")

            # Level 3: Return raw context without generation
            results = hybrid_search.search(query)
            return {
                "answer": "LLM unavailable. Here are relevant docs:",
                "sources": results,
                "metadata": {"fallback": "retrieval_only"}
            }
```

---

## Streaming Responses

### Server-Sent Events (SSE)

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def stream_response(query: str):
    """Stream LLM response token by token."""
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    chain = rag_prompt | llm | StrOutputParser()

    for chunk in chain.stream({"query": query, "context": context}):
        yield f"data: {chunk}\n\n"
```

---

## Implementation Files

- `src/query/__init__.py` - Query module
- `src/query/engine.py` - Main query engine
- `src/query/validator.py` - Query validation
- `src/query/prompts.py` - Prompt templates
- `src/query/chain.py` - LangChain orchestration

---

## Next Steps

Proceed to **Step 7: Evaluation Pipeline (RAGAs)** to implement the automated quality assessment system.
