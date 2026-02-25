"""Prometheus metrics for RAG system."""

from prometheus_client import Counter, Histogram, Gauge, Info

# Query metrics
QUERY_COUNTER = Counter(
    'rag_queries_total',
    'Total number of queries processed',
    ['status']  # success, error, validation_failed, no_results
)

QUERY_LATENCY = Histogram(
    'rag_query_latency_seconds',
    'Query processing latency in seconds',
    ['stage'],  # retrieval, reranking, generation, total
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)

ACTIVE_QUERIES = Gauge(
    'rag_active_queries',
    'Number of queries currently being processed'
)

# Token and cost metrics
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

# Retrieval metrics
RETRIEVAL_RESULTS = Histogram(
    'rag_retrieval_results_count',
    'Number of documents retrieved per query',
    buckets=[0, 1, 3, 5, 10, 20, 50]
)

RERANK_SCORE = Histogram(
    'rag_rerank_score',
    'Re-ranking scores for top results',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# External API metrics
PINECONE_QUERIES = Counter(
    'rag_pinecone_queries_total',
    'Total Pinecone queries'
)

OPENAI_API_CALLS = Counter(
    'rag_openai_api_calls_total',
    'Total OpenAI API calls',
    ['type']  # embedding, completion
)

OPENAI_API_ERRORS = Counter(
    'rag_openai_api_errors_total',
    'Total OpenAI API errors',
    ['error_type']
)

# System info
SYSTEM_INFO = Info(
    'rag_system',
    'RAG system information'
)
