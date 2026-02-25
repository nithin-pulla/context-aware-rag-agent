# Quick Start Guide

Get up and running with the Context-Aware RAG Agent in minutes.

## Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone API key (free tier available)

## Installation

### 1. Clone and Setup

```bash
cd "context-aware-rag-agent for tech doc"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your favorite editor
```

Required keys:
```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1-aws
```

### 3. Create Pinecone Index

```python
from src.retrieval.pinecone_client import PineconeClient

client = PineconeClient()
client.create_index()
```

Or use Python directly:
```bash
python -c "from src.retrieval.pinecone_client import PineconeClient; PineconeClient().create_index()"
```

## Basic Usage

### Step 1: Ingest Documents

Place your documentation files in `data/raw/`:

```bash
mkdir -p data/raw
# Add your .md, .pdf, or .html files to data/raw/
```

Run ingestion:

```bash
python scripts/ingest.py --input data/raw --index
```

This will:
- Parse documents
- Chunk them into manageable pieces
- Generate embeddings
- Index to Pinecone

### Step 2: Query the System

```bash
python scripts/query.py "How do I authenticate with the API?"
```

With cost tracking:
```bash
python scripts/query.py "Your question here" --track-cost
```

### Step 3: Evaluate Quality

```bash
python scripts/evaluate.py
```

This runs the golden set evaluation using RAGAs metrics.

## Python API Usage

```python
from src.query.engine import RAGQueryEngine

# Initialize engine
engine = RAGQueryEngine()

# Query
response = engine.query("How do I configure authentication?")

# Access results
print(response['answer'])
print(f"Sources: {len(response['sources'])}")
```

## Common Tasks

### Add New Documents

```bash
# Copy new docs to data/raw/
python scripts/ingest.py --input data/raw --index
```

### Test Different Chunking Strategies

```bash
# Recursive (default)
python scripts/ingest.py --strategy recursive

# Semantic chunking
python scripts/ingest.py --strategy semantic

# Markdown-aware
python scripts/ingest.py --strategy markdown
```

### Tune Retrieval Parameters

```python
from src.query.engine import RAGQueryEngine

engine = RAGQueryEngine()

# More retrieval, aggressive re-ranking
response = engine.query(
    "your query",
    top_k=10,      # Retrieve 10 docs
    rerank_top_n=3  # Keep top 3 after re-ranking
)
```

### Create Golden Set QA Pairs

```python
from src.dataset.golden_set import GoldenSetManager

manager = GoldenSetManager(Path("data"))

# Add QA pair
qa_pair = manager.add_qa_pair(
    query="How do I install the CLI?",
    answer="Use npm: npm install -g our-cli",
    contexts=["doc_123:chunk_5"],
    metadata={"query_type": "procedural", "difficulty": "simple"},
    sources=[{"doc_id": "doc_123", "title": "Installation Guide"}]
)

# Save
manager.save_golden_set([qa_pair], version="v1.0")
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd "context-aware-rag-agent for tech doc"

# Activate virtual environment
source venv/bin/activate
```

### Pinecone Connection Issues

```bash
# Verify API key is set
python -c "import os; print(os.getenv('PINECONE_API_KEY'))"

# Check index exists
python -c "from src.retrieval.pinecone_client import PineconeClient; print(PineconeClient().get_stats())"
```

### OpenAI Rate Limits

If you hit rate limits, adjust the batch size:

```python
# In config.py or .env
BATCH_SIZE=5  # Reduce from default 10
```

### No Results Returned

Ensure documents are indexed:

```bash
python -c "from src.retrieval.pinecone_client import PineconeClient; print(PineconeClient().get_stats())"
```

Should show `total_vector_count > 0`.

## Next Steps

- Read the [Architecture Documentation](docs/02_system_architecture.md)
- Explore [Evaluation Metrics](docs/07_evaluation_pipeline.md)
- Set up [Observability](docs/08_observability_metrics.md)
- Tune [Hybrid Retrieval](docs/05_hybrid_retrieval.md)

## Example Workflow

```bash
# 1. Ingest sample documents
echo "# API Authentication\nUse OAuth 2.0 with Bearer tokens." > data/raw/auth.md
python scripts/ingest.py --index

# 2. Query
python scripts/query.py "How do I authenticate?" --track-cost

# 3. Evaluate (requires golden set)
python scripts/evaluate.py

# 4. View metrics (if using FastAPI)
curl http://localhost:8000/metrics
```

## Support

For issues or questions:
- Check [method_execution_roadmap.md](method_execution_roadmap.md)
- Review detailed docs in `docs/` directory
- Examine code examples in `src/`
