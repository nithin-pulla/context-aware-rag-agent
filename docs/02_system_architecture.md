# Step 2: High-Level System Architecture & Component Decoupling

## System Overview

The RAG system is decomposed into three primary services with clear boundaries and responsibilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway / Load Balancer              │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼─────────┐      ┌───────▼─────────────┐
│  Query Engine   │      │ Ingestion Service   │
│   (Sync/RT)     │      │    (Async/Batch)    │
└───────┬─────────┘      └──────────┬──────────┘
        │                           │
        │  ┌────────────────────────┘
        │  │
┌───────▼──▼──────────────────────────────────┐
│          Pinecone Vector Database            │
│     (Dense Embeddings + BM25 Sparse)         │
└──────────────────────────────────────────────┘
        │
┌───────▼──────────────┐      ┌────────────────┐
│   OpenAI APIs        │      │  LangSmith     │
│ - Embeddings (Dense) │      │ (Observability)│
│ - LLM (GPT-4 Turbo) │      └────────────────┘
└──────────────────────┘
        │
┌───────▼──────────────────────────┐
│   Evaluation Pipeline (Offline)  │
│        RAGAs Metrics             │
└──────────────────────────────────┘
```

## Component Architecture

### 1. Ingestion Service (Async/Batch)

**Responsibility**: Transform raw documents into searchable vectors and store in Pinecone.

**Subcomponents**:
```
Document Loader → Chunker → Embedder → Indexer
```

**Interfaces**:
- **Input**: Raw files (PDF, Markdown, HTML, reStructuredText)
- **Output**: Indexed vectors in Pinecone + metadata store

**Communication Protocol**:
- **Trigger**: CLI, scheduled cron, or message queue (for production)
- **State Management**: Checkpointing to file system or Redis
- **Error Handling**: Dead letter queue for failed documents

**Key Responsibilities**:
1. **Document Parsing**: Extract text from various formats
2. **Chunking**: Split documents using configurable strategies (RecursiveCharacter, Semantic)
3. **Embedding**: Generate dense embeddings (OpenAI) and BM25 sparse vectors
4. **Indexing**: Upsert to Pinecone with metadata (source, version, timestamp)
5. **Deduplication**: Detect and skip duplicate content

**State Machine**:
```
PENDING → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETED
                ↓         ↓          ↓           ↓
              FAILED    FAILED     FAILED      FAILED
                ↓         ↓          ↓           ↓
              RETRY     RETRY      RETRY       RETRY
                                                  ↓
                                              DEAD_LETTER
```

---

### 2. Query Engine (Sync/Real-Time)

**Responsibility**: Process user queries and generate contextually-aware responses.

**Subcomponents**:
```
Query Parser → Retriever → Re-ranker → Generator → Response Formatter
```

**Interfaces**:
- **Input**: Natural language query (REST API or CLI)
- **Output**: JSON response with answer, sources, and metadata

**Communication Protocol**:
- **API Type**: REST (FastAPI)
- **Endpoints**:
  - `POST /query` - Submit query
  - `GET /health` - Health check
  - `GET /metrics` - Prometheus metrics
- **Timeout**: 30 seconds (configurable)
- **Rate Limiting**: 100 req/min per user (token bucket)

**Key Responsibilities**:
1. **Query Processing**: Parse and normalize user input
2. **Hybrid Retrieval**: Fetch top-k documents using Pinecone (dense + BM25)
3. **Re-ranking**: Cross-encoder scoring for precision
4. **Context Building**: Construct LLM prompt with retrieved chunks
5. **Generation**: Call OpenAI GPT-4 for answer synthesis
6. **Citation Extraction**: Map answer to source documents

**State Machine**:
```
RECEIVED → VALIDATED → RETRIEVING → RE_RANKING → GENERATING → RESPONDING
              ↓            ↓             ↓             ↓            ↓
           INVALID    TIMEOUT      TIMEOUT        TIMEOUT      TIMEOUT
              ↓            ↓             ↓             ↓            ↓
            ERROR      RETRY         RETRY         RETRY        ERROR
                         ↓             ↓             ↓
                    CIRCUIT_OPEN  CIRCUIT_OPEN  CIRCUIT_OPEN
                         ↓             ↓             ↓
                     FALLBACK      FALLBACK      FALLBACK
```

**Fallback Strategy**:
- **Retrieval Failure**: Return cached results or degraded search
- **LLM Failure**: Return raw retrieved context without synthesis
- **Timeout**: Partial response with streaming tokens

---

### 3. Evaluation Pipeline (Offline/Batch)

**Responsibility**: Measure system quality using RAGAs metrics and golden set.

**Subcomponents**:
```
Golden Set Loader → Query Executor → RAGAs Evaluator → Metrics Reporter
```

**Interfaces**:
- **Input**: Golden QA pairs (JSON)
- **Output**: Evaluation report (JSON, CSV, HTML dashboard)

**Communication Protocol**:
- **Trigger**: CI/CD pipeline, scheduled nightly runs
- **Execution**: Batch processing (10 queries at a time)
- **Storage**: Results stored in time-series database or S3

**Key Responsibilities**:
1. **Golden Set Management**: Load curated QA pairs
2. **Query Execution**: Run queries through Query Engine
3. **RAGAs Evaluation**: Compute Faithfulness, Relevance, Precision, Recall
4. **Threshold Gating**: Fail CI/CD if metrics below baseline
5. **Regression Detection**: Compare with historical metrics

**Metrics Computed**:
- **Faithfulness**: Answer grounded in retrieved context
- **Answer Relevance**: Response quality for query
- **Context Precision**: Irrelevant docs in top-k
- **Context Recall**: Coverage of necessary info

---

## Data Flow State Machines

### Ingestion Pipeline Flow

```
┌──────────────┐
│   Raw Docs   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Loader     │ (Parse PDF/MD/HTML)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Chunker    │ (RecursiveCharacter / Semantic)
└──────┬───────┘
       │
       ▼ (Parallel)
┌──────────────────────────────┐
│  Dense Embedder (OpenAI)     │
│  Sparse Embedder (BM25)      │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────┐
│   Pinecone   │ (Upsert with metadata)
└──────────────┘
```

### Query Processing Flow

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Validator  │ (Check length, sanitize)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│   Hybrid Retrieval           │
│   - Dense: text-embedding-3  │
│   - Sparse: BM25             │
│   - Alpha: 0.5 (tunable)     │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────┐
│   Re-ranker  │ (Cross-encoder for top-3)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Prompt Build │ (Template + context + query)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  GPT-4 Call  │ (Stream tokens)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Response   │ (Answer + sources + metadata)
└──────────────┘
```

## Inter-Service Communication Protocols

### Synchronous (Query Engine ↔ External APIs)

**Protocol**: HTTPS REST

**Services**:
- **Pinecone**: Vector search (POST /query)
- **OpenAI**: Embeddings (POST /embeddings), Completions (POST /chat/completions)

**Error Handling**:
- **Retries**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Circuit Breaker**: Open after 5 consecutive failures, half-open after 30s
- **Timeout**: 10s for embeddings, 30s for LLM generation

### Asynchronous (Ingestion Service)

**Protocol**: Message Queue (future: SQS/RabbitMQ) or Direct File System

**Pattern**: Producer-Consumer with idempotent processing

**Checkpointing**: Store processed document IDs in Redis or local DB

---

## Deployment Architecture (Future Production)

```
┌─────────────────────────────────────────────┐
│          Load Balancer (ALB/NGINX)          │
└────────┬──────────────────────────┬─────────┘
         │                          │
┌────────▼─────────┐       ┌────────▼─────────┐
│  Query Engine    │       │  Query Engine    │
│   (Container 1)  │       │   (Container 2)  │
└────────┬─────────┘       └────────┬─────────┘
         │                          │
         └──────────┬───────────────┘
                    │
         ┌──────────▼─────────────┐
         │   Pinecone Serverless  │
         └────────────────────────┘
```

**Scaling Strategy**:
- **Horizontal**: Auto-scale Query Engine containers (K8s HPA)
- **Caching**: Redis for semantic cache (query → response)
- **Database**: Pinecone serverless (auto-scales with demand)

---

## Technology Stack Summary

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **API Framework** | FastAPI | Async support, auto-docs, Pydantic validation |
| **Orchestration** | LangChain | Abstractions for chains, callbacks, and memory |
| **Vector DB** | Pinecone | Serverless, hybrid search, low latency |
| **Embeddings** | OpenAI text-embedding-3-small | High quality, cost-effective |
| **LLM** | GPT-4 Turbo | Best reasoning, context window (128k tokens) |
| **Evaluation** | RAGAs | Industry-standard RAG metrics |
| **Observability** | LangSmith + Prometheus | LLM tracing + system metrics |
| **Deployment** | Docker + Kubernetes (future) | Portability, auto-scaling |

---

## Security & Access Control

**API Security**:
- API keys for authentication (future: OAuth2)
- Rate limiting per user/IP
- Input sanitization to prevent prompt injection

**Data Security**:
- Pinecone API keys stored in environment variables
- No PII in logs or traces
- Document access control via namespaces (future)

---

## Next Steps

Proceed to **Step 3: Dataset Strategy & Golden Set Construction** to establish the baseline QA pairs for evaluation.
