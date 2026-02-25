# Step 8: Metrics, Observability & Cost Engineering

## Overview

Production RAG systems require comprehensive observability to debug latency issues, track costs, and ensure reliability. This step implements telemetry, monitoring, and cost tracking.

## Observability Stack

```
┌─────────────────┐
│  Application    │
│   (RAG System)  │
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐       ┌─────────────────┐
│   LangSmith     │       │  Prometheus     │
│ (LLM Tracing)   │       │   (Metrics)     │
└─────────────────┘       └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │    Grafana      │
                          │  (Dashboards)   │
                          └─────────────────┘
```

---

## Metrics Collection

### Latency Metrics

**Key Metrics**:
1. **TTFT (Time to First Token)**: Time until LLM starts generating
2. **Generation Time**: Total time for LLM response
3. **Retrieval Time**: Time to fetch from Pinecone
4. **Re-ranking Time**: Cross-encoder processing time
5. **End-to-End Time**: Total query processing time

**Implementation**:
```python
import time
from prometheus_client import Histogram, Counter, Gauge

# Define metrics
QUERY_LATENCY = Histogram(
    'rag_query_latency_seconds',
    'Query processing latency',
    ['stage']  # retrieval, reranking, generation, total
)

QUERY_COUNTER = Counter(
    'rag_queries_total',
    'Total number of queries',
    ['status']  # success, error, validation_failed
)

ACTIVE_QUERIES = Gauge(
    'rag_active_queries',
    'Number of queries currently being processed'
)

# Usage
def query_with_metrics(query: str):
    ACTIVE_QUERIES.inc()

    start_time = time.time()

    try:
        # Retrieval
        retrieval_start = time.time()
        results = retriever.search(query)
        QUERY_LATENCY.labels(stage='retrieval').observe(time.time() - retrieval_start)

        # Re-ranking
        rerank_start = time.time()
        results = reranker.rerank(query, results)
        QUERY_LATENCY.labels(stage='reranking').observe(time.time() - rerank_start)

        # Generation
        gen_start = time.time()
        answer = llm.generate(query, results)
        QUERY_LATENCY.labels(stage='generation').observe(time.time() - gen_start)

        # Total
        QUERY_LATENCY.labels(stage='total').observe(time.time() - start_time)
        QUERY_COUNTER.labels(status='success').inc()

        return answer

    except Exception as e:
        QUERY_COUNTER.labels(status='error').inc()
        raise

    finally:
        ACTIVE_QUERIES.dec()
```

---

### Cost Tracking

**Components**:
1. **Embedding Costs**: OpenAI text-embedding-3-small
2. **LLM Generation Costs**: GPT-4 Turbo input + output tokens
3. **Pinecone Costs**: Queries and storage

**Pricing** (as of 2024):
- Embeddings: $0.00002 per 1K tokens
- GPT-4 Turbo: $0.01 per 1K input tokens, $0.03 per 1K output tokens
- Pinecone Serverless: $0.30 per 1M read units

**Implementation**:
```python
import tiktoken
from prometheus_client import Counter

# Cost counters
EMBEDDING_TOKENS = Counter(
    'rag_embedding_tokens_total',
    'Total embedding tokens processed'
)

LLM_INPUT_TOKENS = Counter(
    'rag_llm_input_tokens_total',
    'Total LLM input tokens'
)

LLM_OUTPUT_TOKENS = Counter(
    'rag_llm_output_tokens_total',
    'Total LLM output tokens'
)

PINECONE_QUERIES = Counter(
    'rag_pinecone_queries_total',
    'Total Pinecone queries'
)

# Cost computation
def compute_query_cost(metrics: dict) -> float:
    """Compute cost for a single query."""
    # Embedding cost
    embedding_cost = (metrics['embedding_tokens'] / 1000) * 0.00002

    # LLM cost
    input_cost = (metrics['llm_input_tokens'] / 1000) * 0.01
    output_cost = (metrics['llm_output_tokens'] / 1000) * 0.03
    llm_cost = input_cost + output_cost

    # Pinecone cost (approximate)
    pinecone_cost = metrics['pinecone_queries'] * 0.0000003

    return embedding_cost + llm_cost + pinecone_cost

# Track costs
def track_query_costs(query: str, response: str, contexts: list):
    """Track costs for a query."""
    encoding = tiktoken.encoding_for_model("gpt-4")

    # Embedding tokens (query only, contexts already embedded)
    query_tokens = len(encoding.encode(query))
    EMBEDDING_TOKENS.inc(query_tokens)

    # LLM input tokens (prompt + context)
    input_text = query + "\n".join([c['text'] for c in contexts])
    input_tokens = len(encoding.encode(input_text))
    LLM_INPUT_TOKENS.inc(input_tokens)

    # LLM output tokens
    output_tokens = len(encoding.encode(response))
    LLM_OUTPUT_TOKENS.inc(output_tokens)

    # Pinecone queries
    PINECONE_QUERIES.inc(1)

    # Compute and log cost
    cost = compute_query_cost({
        'embedding_tokens': query_tokens,
        'llm_input_tokens': input_tokens,
        'llm_output_tokens': output_tokens,
        'pinecone_queries': 1
    })

    logger.info(f"Query cost: ${cost:.4f}")
    return cost
```

---

## LangSmith Integration

**Setup**:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "rag-tech-docs"
```

**Features**:
- **Trace Visualization**: See entire chain execution
- **Latency Breakdown**: Identify bottlenecks
- **LLM Call Inspection**: View prompts and responses
- **Error Tracking**: Debug failed queries

**Custom Callbacks**:
```python
from langchain.callbacks import LangChainTracer

tracer = LangChainTracer(
    project_name="rag-tech-docs",
    tags=["production", "hybrid-search"]
)

# Use in chain
chain.invoke(
    {"query": query},
    config={"callbacks": [tracer]}
)
```

---

## Prometheus Metrics Export

### FastAPI Integration

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()

# Mount prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/query")
async def query_endpoint(q: str):
    """Query endpoint with metrics."""
    with QUERY_LATENCY.labels(stage='total').time():
        result = query_engine.query(q)
        QUERY_COUNTER.labels(status='success').inc()
        return result
```

### Metrics Endpoint

```
GET /metrics

# HELP rag_query_latency_seconds Query processing latency
# TYPE rag_query_latency_seconds histogram
rag_query_latency_seconds_bucket{stage="retrieval",le="0.1"} 245
rag_query_latency_seconds_bucket{stage="retrieval",le="0.5"} 1032
rag_query_latency_seconds_sum{stage="retrieval"} 234.5
rag_query_latency_seconds_count{stage="retrieval"} 1500

# HELP rag_queries_total Total number of queries
# TYPE rag_queries_total counter
rag_queries_total{status="success"} 1450
rag_queries_total{status="error"} 50
```

---

## Grafana Dashboards

### Key Panels

**1. Query Volume**
```promql
rate(rag_queries_total[5m])
```

**2. P95 Latency**
```promql
histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))
```

**3. Error Rate**
```promql
rate(rag_queries_total{status="error"}[5m]) /
rate(rag_queries_total[5m])
```

**4. Cost Per Hour**
```promql
(
  rate(rag_embedding_tokens_total[1h]) / 1000 * 0.00002 +
  rate(rag_llm_input_tokens_total[1h]) / 1000 * 0.01 +
  rate(rag_llm_output_tokens_total[1h]) / 1000 * 0.03
)
```

---

## Circuit Breaking & Rate Limiting

### Token Bucket Rate Limiter

```python
import time
from threading import Lock

class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns:
            True if tokens available, False otherwise
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

# Usage
rate_limiter = TokenBucket(rate=10, capacity=100)  # 10 req/sec

def rate_limited_query(query: str):
    """Query with rate limiting."""
    if not rate_limiter.consume():
        raise Exception("Rate limit exceeded")

    return query_engine.query(query)
```

---

## Alerting Rules

### Prometheus Alerts

```yaml
# alerts.yml
groups:
  - name: rag_system
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          rate(rag_queries_total{status="error"}[5m]) /
          rate(rag_queries_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            rate(rag_query_latency_seconds_bucket{stage="total"}[5m])
          ) > 20
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency detected"
          description: "P99 latency is {{ $value }}s"

      - alert: HighCost
        expr: |
          (
            rate(rag_llm_input_tokens_total[1h]) / 1000 * 0.01 +
            rate(rag_llm_output_tokens_total[1h]) / 1000 * 0.03
          ) > 10
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "High hourly cost detected"
          description: "Hourly cost is ${{ $value }}"
```

---

## Distributed Tracing (OpenTelemetry)

### Trace Context Propagation

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add exporter
span_processor = BatchSpanProcessor(OTLPSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument query
def query_with_tracing(query: str):
    """Query with distributed tracing."""
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query.text", query[:100])

        # Retrieval span
        with tracer.start_as_current_span("retrieval"):
            results = retriever.search(query)
            span.set_attribute("retrieval.count", len(results))

        # Generation span
        with tracer.start_as_current_span("generation"):
            answer = llm.generate(query, results)
            span.set_attribute("generation.length", len(answer))

        return answer
```

---

## Implementation Files

- `src/observability/__init__.py` - Observability module
- `src/observability/metrics.py` - Prometheus metrics
- `src/observability/tracing.py` - Distributed tracing
- `src/observability/cost_tracker.py` - Cost tracking
- `configs/prometheus.yml` - Prometheus configuration
- `configs/grafana_dashboard.json` - Grafana dashboard

---

## Summary

This completes the implementation of all 8 phases of the production-grade RAG system:

✅ **Step 1**: Problem formulation and portfolio positioning
✅ **Step 2**: High-level system architecture
✅ **Step 3**: Dataset strategy and golden set construction
✅ **Step 4**: Data ingestion and chunking framework
✅ **Step 5**: Hybrid retrieval strategy (Pinecone)
✅ **Step 6**: Query engine and LangChain orchestration
✅ **Step 7**: Evaluation pipeline (RAGAs)
✅ **Step 8**: Metrics, observability, and cost engineering

The system is now production-ready with comprehensive monitoring, evaluation, and cost tracking capabilities.
