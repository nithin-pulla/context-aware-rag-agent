# Step 5: Hybrid Retrieval Strategy (Pinecone)

## Overview

Hybrid search combines dense semantic vectors with sparse keyword-based retrieval (BM25) to leverage the strengths of both approaches, resulting in superior retrieval accuracy.

## Hybrid Search Architecture

```
┌─────────────┐
│    Query    │
└──────┬──────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
┌──────────────┐       ┌──────────────┐
│Dense Embed   │       │Sparse Embed  │
│(Semantic)    │       │(BM25)        │
└──────┬───────┘       └──────┬───────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────────┐
│       Pinecone Hybrid Search         │
│   Score = α·dense + (1-α)·sparse     │
└──────────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │   Top-K      │
        │  Results     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Re-ranker    │
        │(Cross-Encoder│
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  Top-N Final │
        └──────────────┘
```

## Dense vs. Sparse Retrieval

### Dense Embeddings (Semantic Search)

**Model**: `text-embedding-3-small` (1536 dimensions)

**Strengths**:
- Captures semantic meaning
- Handles synonyms and paraphrases
- Robust to typos
- Excels at conceptual queries

**Weaknesses**:
- May miss exact keyword matches
- Computationally expensive
- Requires quality embeddings

**Example Query**: "How do I secure my API?"
- Retrieves docs about "authentication", "authorization", "security best practices"

---

### Sparse Embeddings (BM25)

**Algorithm**: BM25 (Best Matching 25) - probabilistic relevance framework

**Strengths**:
- Exact keyword matching
- No embedding required (term frequency based)
- Fast and interpretable
- Excels at factual/specific queries

**Weaknesses**:
- No semantic understanding
- Sensitive to typos
- Struggles with synonyms

**Example Query**: "API rate limit 429 error"
- Retrieves docs containing exact terms "API", "rate limit", "429"

---

## Convex Combination (Alpha Tuning)

### Scoring Formula

```
final_score = α * dense_score + (1 - α) * sparse_score

where α ∈ [0, 1]
```

### Alpha Values

| Alpha | Emphasis | Use Case |
|-------|----------|----------|
| **α = 0.0** | 100% BM25 | Pure keyword search |
| **α = 0.3** | 70% BM25, 30% semantic | Factual queries |
| **α = 0.5** | Balanced (default) | General-purpose |
| **α = 0.7** | 70% semantic, 30% BM25 | Conceptual queries |
| **α = 1.0** | 100% semantic | Pure semantic search |

### Tuning Strategy

**Approach**: Grid search over α values using golden set

```python
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for alpha in alphas:
    results = evaluate_golden_set(alpha=alpha)
    metrics = compute_ragas_metrics(results)
    print(f"Alpha={alpha}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
```

**Optimal α**: Typically 0.4-0.6 for technical docs (balanced)

---

## Pinecone Index Configuration

### Index Creation

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create serverless index
pc.create_index(
    name="rag-tech-docs",
    dimension=1536,  # text-embedding-3-small
    metric="dotproduct",  # For normalized vectors
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

### Upsert with Hybrid Vectors

```python
index = pc.Index("rag-tech-docs")

vectors = []
for chunk in chunks:
    vectors.append({
        'id': chunk['chunk_id'],
        'values': chunk['dense_embedding'],  # Dense vector
        'sparse_values': {
            'indices': chunk['sparse_embedding']['indices'],
            'values': chunk['sparse_embedding']['values']
        },
        'metadata': {
            'text': chunk['text'],
            'source': chunk['source'],
            'chunk_index': chunk['chunk_index']
        }
    })

# Upsert in batches
index.upsert(vectors=vectors, namespace="default")
```

---

## Query Processing

### Hybrid Query

```python
def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 10):
    # Generate embeddings
    dense_vector, sparse_vector = embedder.embed_query(query)

    # Query Pinecone
    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
        alpha=alpha  # Hybrid search weight
    )

    return results['matches']
```

---

## Cross-Encoder Re-ranking

### Purpose

Re-rank top-k results using a cross-encoder model that directly computes query-document relevance.

**Why Re-rank?**
- Bi-encoders (dense embeddings) are fast but less accurate
- Cross-encoders are slow but highly accurate
- Compromise: Retrieve with bi-encoder, re-rank with cross-encoder

### Implementation

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query: str, results: list, top_n: int = 3):
    # Prepare query-document pairs
    pairs = [(query, result['metadata']['text']) for result in results]

    # Compute cross-encoder scores
    scores = cross_encoder.predict(pairs)

    # Sort by score
    ranked_results = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [result for result, score in ranked_results[:top_n]]
```

### Performance Impact

| Stage | Latency | Accuracy |
|-------|---------|----------|
| **Bi-encoder (top-10)** | 50ms | 0.75 Precision@10 |
| **+ Re-ranking (top-3)** | +150ms | 0.90 Precision@3 |

**Trade-off**: 3x latency for 20% accuracy gain (worthwhile for quality-critical applications)

---

## Retrieval Metrics

### Primary Metrics

**1. Precision@k**: Fraction of retrieved docs that are relevant
```
Precision@k = (Relevant docs in top-k) / k
```

**2. Recall@k**: Fraction of all relevant docs retrieved
```
Recall@k = (Relevant docs in top-k) / (Total relevant docs)
```

**3. Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant doc
```
MRR = (1/N) * Σ(1 / rank_i)
```

**4. Normalized Discounted Cumulative Gain (NDCG@k)**: Weighted relevance score
```
DCG@k = Σ(rel_i / log₂(i + 1))
NDCG@k = DCG@k / IDCG@k
```

### RAGAs Context Metrics

**1. Context Precision**: Measures whether retrieved contexts are relevant
```python
# Computed by RAGAs
context_precision = evaluate_context_precision(query, retrieved_contexts, ground_truth_answer)
```

**2. Context Recall**: Measures whether all necessary info is retrieved
```python
context_recall = evaluate_context_recall(ground_truth_contexts, retrieved_contexts)
```

---

## Retrieval Optimization Strategies

### 1. Namespace Partitioning

Segment index by product, version, or doc type:

```python
# Query specific namespace
results = index.query(
    vector=dense_vector,
    namespace="product-v2",  # Only search within this namespace
    top_k=10
)
```

**Benefits**:
- Faster queries (smaller search space)
- Isolated updates (no index rebuild)
- Multi-tenancy support

---

### 2. Metadata Filtering

Filter results by metadata:

```python
results = index.query(
    vector=dense_vector,
    filter={
        "doc_type": {"$eq": "api-reference"},
        "version": {"$gte": "2.0"}
    },
    top_k=10
)
```

---

### 3. Query Expansion

Enhance queries with synonyms or related terms:

```python
def expand_query(query: str, llm) -> str:
    prompt = f"Generate 3 related search terms for: {query}"
    expansion = llm.invoke(prompt)
    return f"{query} {expansion}"
```

---

### 4. Reciprocal Rank Fusion (RRF)

Combine multiple retrieval strategies:

```python
def reciprocal_rank_fusion(rankings: list, k: int = 60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

---

## Implementation Files

- `src/retrieval/__init__.py` - Retrieval module
- `src/retrieval/hybrid_search.py` - Hybrid search logic
- `src/retrieval/reranker.py` - Cross-encoder re-ranking
- `src/retrieval/pinecone_client.py` - Pinecone integration

---

## Next Steps

Proceed to **Step 6: Query Engine & LangChain Orchestration** to build the end-to-end query processing pipeline.
