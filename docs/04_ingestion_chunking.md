# Step 4: Data Ingestion & Chunking Experimentation Framework

## Overview

The ingestion pipeline transforms raw documentation into searchable vector embeddings. The chunking strategy directly impacts retrieval quality and must be systematically evaluated.

## Pipeline Architecture

```
┌─────────────┐
│  Raw Docs   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Loader    │ (Parse PDF/MD/HTML)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Chunker    │ (RecursiveCharacter / Semantic)
└──────┬──────┘
       │
       ├─────────────────┐
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│Dense Embed  │   │Sparse Embed │
│(OpenAI)     │   │(BM25)       │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                ▼
       ┌─────────────────┐
       │  Pinecone Index │
       └─────────────────┘
```

## Chunking Strategies

### Strategy 1: Recursive Character Splitter (Baseline)

**Algorithm**: Split on delimiters hierarchically

**Delimiters** (in order):
1. `\n\n` (paragraph breaks)
2. `\n` (line breaks)
3. `. ` (sentence ends)
4. ` ` (word boundaries)

**Parameters**:
- `chunk_size`: 512 tokens (configurable: 256, 512, 1024)
- `chunk_overlap`: 50 tokens (10% overlap)

**Pros**:
- Simple, deterministic
- Preserves natural language boundaries
- Fast processing

**Cons**:
- May split semantic units (code blocks, lists)
- Fixed size doesn't adapt to content

**Implementation**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_text(document_text)
```

---

### Strategy 2: Semantic Chunking (Advanced)

**Algorithm**: Split based on semantic similarity between sentences

**Process**:
1. Split document into sentences
2. Compute embeddings for each sentence
3. Merge sentences while cosine similarity > threshold (e.g., 0.7)
4. Create variable-size chunks

**Parameters**:
- `similarity_threshold`: 0.7
- `min_chunk_size`: 100 tokens
- `max_chunk_size`: 1024 tokens

**Pros**:
- Preserves semantic coherence
- Adapts to content structure
- Better for conceptual queries

**Cons**:
- Slower (requires embeddings)
- Variable chunk sizes complicate comparison
- May create very small or large chunks

**Implementation**:
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
chunks = splitter.create_documents([document_text])
```

---

### Strategy 3: Markdown-Aware Splitting (Domain-Specific)

**Algorithm**: Respect Markdown structure (headings, code blocks, lists)

**Rules**:
- Never split code blocks
- Keep lists together
- Include heading as context for each chunk
- Preserve links and references

**Parameters**:
- `chunk_size`: 512 tokens
- `headers_to_split_on`: [("#", "h1"), ("##", "h2"), ("###", "h3")]

**Pros**:
- Preserves document structure
- Context-aware (includes headings)
- Ideal for technical docs

**Cons**:
- Only works for Markdown
- More complex logic

**Implementation**:
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
chunks = splitter.split_text(markdown_text)
```

---

## Chunking Evaluation Framework

### Metrics

**1. Retrieval Accuracy (Primary)**
- Test queries against golden set
- Measure: Precision@k, Recall@k, MRR (Mean Reciprocal Rank)

**2. Chunk Quality (Secondary)**
- **Coherence**: Measure semantic similarity within chunk
- **Completeness**: Check if chunks contain complete thoughts
- **Diversity**: Avoid redundant information across chunks

**3. Operational Metrics**
- Processing time per document
- Total chunks generated
- Chunk size distribution

### Experiment Design

**Hypothesis**: Markdown-aware chunking outperforms recursive splitting for technical docs.

**Variables**:
- **Independent**: Chunking strategy (Recursive, Semantic, Markdown)
- **Dependent**: Context Precision, Context Recall (RAGAs metrics)

**Control**:
- Same embedding model (text-embedding-3-small)
- Same retrieval parameters (top_k=5, alpha=0.5)
- Same golden set (50 QA pairs)

**Procedure**:
1. Chunk corpus with each strategy
2. Index chunks in Pinecone (separate namespaces)
3. Run golden set queries
4. Compute RAGAs metrics
5. Compare distributions (t-test for significance)

---

## Document Ingestion Pipeline

### ETL Stages

#### Stage 1: Document Loading
```python
from pathlib import Path
from src.dataset.parser import DocumentParser

def load_documents(data_dir: Path):
    parser = DocumentParser()
    documents = []

    for file_path in data_dir.rglob('*'):
        if file_path.is_file():
            parsed = parser.parse(file_path)
            if parsed:
                documents.append(parsed)

    return documents
```

#### Stage 2: Chunking
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = []
    for doc in documents:
        text = doc['full_text']
        doc_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                'chunk_id': f"{doc['metadata']['content_hash']}:chunk_{i}",
                'text': chunk_text,
                'source': doc['source'],
                'chunk_index': i,
                'metadata': doc['metadata']
            })

    return chunks
```

#### Stage 3: Embedding Generation
```python
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder

def generate_embeddings(chunks):
    # Dense embeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c['text'] for c in chunks]
    dense_vectors = embeddings_model.embed_documents(texts)

    # Sparse embeddings (BM25)
    bm25 = BM25Encoder()
    bm25.fit(texts)
    sparse_vectors = bm25.encode_documents(texts)

    # Attach to chunks
    for i, chunk in enumerate(chunks):
        chunk['dense_embedding'] = dense_vectors[i]
        chunk['sparse_embedding'] = sparse_vectors[i]

    return chunks
```

#### Stage 4: Indexing to Pinecone
```python
from pinecone import Pinecone

def index_chunks(chunks, index_name):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        vectors = []
        for chunk in batch:
            vectors.append({
                'id': chunk['chunk_id'],
                'values': chunk['dense_embedding'],
                'sparse_values': chunk['sparse_embedding'],
                'metadata': {
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index']
                }
            })

        index.upsert(vectors=vectors)
```

---

## Checkpointing & Idempotency

### Checkpoint Strategy

Store processed document hashes to skip on re-runs:

```python
import json
from pathlib import Path

class IngestionCheckpoint:
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.processed = self._load()

    def _load(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()

    def is_processed(self, doc_hash: str) -> bool:
        return doc_hash in self.processed

    def mark_processed(self, doc_hash: str):
        self.processed.add(doc_hash)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed), f)
```

---

## Error Handling

### Dead Letter Queue

Failed documents stored for manual review:

```python
import jsonlines

class DeadLetterQueue:
    def __init__(self, dlq_file: Path):
        self.dlq_file = dlq_file

    def add(self, doc: dict, error: Exception):
        with jsonlines.open(self.dlq_file, mode='a') as writer:
            writer.write({
                'source': doc['source'],
                'error': str(error),
                'timestamp': datetime.now().isoformat()
            })
```

---

## Implementation Files

- `src/ingestion/__init__.py` - Ingestion module
- `src/ingestion/chunker.py` - Chunking strategies
- `src/ingestion/embedder.py` - Embedding generation
- `src/ingestion/indexer.py` - Pinecone indexing
- `src/ingestion/pipeline.py` - End-to-end pipeline

---

## Next Steps

Proceed to **Step 5: Hybrid Retrieval Strategy (Pinecone)** to implement the query-time retrieval system.
