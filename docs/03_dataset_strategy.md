# Step 3: Dataset Strategy & Golden Set Construction

## Overview

The Golden Set is a curated collection of question-answer pairs with ground truth answers and source citations. It serves as an immutable baseline for regression testing and evaluation of the RAG system.

## Dataset Composition

### Golden Set Requirements

**Size**: 100-500 QA pairs minimum for statistical significance

**Coverage**:
- **Query Types**: Factual, procedural, conceptual, troubleshooting
- **Difficulty Levels**: Simple (direct lookup), Medium (multi-hop), Complex (reasoning)
- **Document Types**: API docs, tutorials, troubleshooting guides, architectural overviews

**Statistical Composition**:
```
┌─────────────────────────────────────────┐
│ Query Type Distribution                 │
├─────────────────────────────────────────┤
│ Factual (What/When/Who):      40%      │
│ Procedural (How-to):          30%      │
│ Conceptual (Why/Explain):     20%      │
│ Troubleshooting (Debug):      10%      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Difficulty Distribution                 │
├─────────────────────────────────────────┤
│ Simple (1 chunk):             50%      │
│ Medium (2-3 chunks):          35%      │
│ Complex (4+ chunks):          15%      │
└─────────────────────────────────────────┘
```

## Golden Set Schema

```json
{
  "id": "qa_001",
  "query": "How do I authenticate with the REST API?",
  "ground_truth_answer": "The REST API uses OAuth 2.0 for authentication. Include the Authorization header with 'Bearer <token>' in each request.",
  "ground_truth_contexts": [
    "doc_id_123:chunk_5",
    "doc_id_456:chunk_12"
  ],
  "metadata": {
    "query_type": "procedural",
    "difficulty": "simple",
    "topic": "authentication",
    "created_at": "2024-01-15",
    "validator": "human"
  },
  "expected_sources": [
    {
      "doc_id": "doc_id_123",
      "title": "API Authentication Guide",
      "url": "https://docs.example.com/auth"
    }
  ]
}
```

## Golden Set Construction Pipeline

### Approach 1: Manual Curation (High Quality, Low Scale)

**Process**:
1. Domain experts review documentation
2. Identify common user questions
3. Write ground truth answers with citations
4. Peer review for accuracy

**Pros**: Highest quality, domain-specific
**Cons**: Time-intensive, doesn't scale

**Use Case**: Initial 50-100 core QA pairs

---

### Approach 2: Synthetic Generation (LLM-as-a-Judge)

**Process**:
```
Document Chunk → LLM (GPT-4) → Generate Question → Human Validation
```

**Prompt Template**:
```
You are a technical documentation expert. Given the following documentation chunk, generate a realistic user question that can be answered using this content.

Documentation:
{chunk_text}

Generate:
1. A natural user question
2. The direct answer (2-3 sentences)
3. Query type (factual/procedural/conceptual/troubleshooting)
4. Difficulty (simple/medium/complex)

Format as JSON.
```

**Quality Filters**:
- Remove generic questions ("What is this about?")
- Deduplicate similar questions (cosine similarity > 0.85)
- Human spot-check 10% of generated pairs

**Pros**: Scales to thousands of QA pairs
**Cons**: May lack creativity, requires validation

**Use Case**: Expanding to 500+ QA pairs

---

### Approach 3: User Query Mining (Real-World Distribution)

**Process**:
1. Collect anonymized user queries from production logs
2. Sample top 100 frequent queries
3. Domain experts provide ground truth answers
4. Map to source documents

**Pros**: Reflects real usage patterns
**Cons**: Requires production deployment first

**Use Case**: Post-launch refinement

---

## Data Parsing Strategy

### Input: Raw Technical Documentation

**Supported Formats**:
- **Markdown** (.md): Most common for developer docs
- **reStructuredText** (.rst): Python ecosystem (Sphinx)
- **HTML** (.html): Web-based documentation
- **PDF** (.pdf): Manuals and whitepapers

### Parsing Logic

#### 1. Markdown Parsing
```python
from markdown import markdown
from bs4 import BeautifulSoup

def parse_markdown(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        md_content = f.read()

    # Convert to HTML for structure extraction
    html = markdown(md_content, extensions=['extra', 'codehilite'])
    soup = BeautifulSoup(html, 'html.parser')

    # Extract sections
    sections = []
    for heading in soup.find_all(['h1', 'h2', 'h3']):
        title = heading.get_text()
        content = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ['h1', 'h2', 'h3']:
                break
            content.append(sibling.get_text())
        sections.append({'title': title, 'content': '\n'.join(content)})

    return {
        'source': file_path,
        'sections': sections,
        'metadata': {'format': 'markdown'}
    }
```

#### 2. PDF Parsing
```python
from pypdf import PdfReader

def parse_pdf(file_path: str) -> dict:
    reader = PdfReader(file_path)
    pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        pages.append({'page_num': page_num + 1, 'text': text})

    return {
        'source': file_path,
        'pages': pages,
        'metadata': {'format': 'pdf', 'total_pages': len(pages)}
    }
```

#### 3. HTML Parsing
```python
from bs4 import BeautifulSoup

def parse_html(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Remove script and style tags
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Extract main content (heuristic: look for common content divs)
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    text = main_content.get_text(separator='\n', strip=True)

    return {
        'source': file_path,
        'text': text,
        'metadata': {'format': 'html'}
    }
```

---

## Deterministic Processing Rules

### Normalization
1. **Whitespace**: Collapse multiple spaces to single space
2. **Encoding**: Convert to UTF-8
3. **Line Endings**: Normalize to \n
4. **Code Blocks**: Preserve formatting with markers

### Metadata Extraction
- **Source URL**: From filename or frontmatter
- **Version**: From git commit hash or document version field
- **Timestamp**: File modification time
- **Author**: From git blame or document metadata

### Deduplication
- Compute MD5 hash of normalized content
- Skip documents with duplicate hashes
- Log duplicate sources for review

---

## Golden Set Storage

### File Format: JSONL (JSON Lines)

**Rationale**: Easy to append, stream, and version control

```jsonl
{"id": "qa_001", "query": "...", "ground_truth_answer": "...", ...}
{"id": "qa_002", "query": "...", "ground_truth_answer": "...", ...}
```

### Directory Structure
```
data/
├── golden_set/
│   ├── v1.0/
│   │   ├── golden_set.jsonl
│   │   ├── metadata.json
│   │   └── changelog.md
│   ├── v1.1/
│   │   └── ...
│   └── current -> v1.1/  (symlink)
├── raw/
│   ├── markdown/
│   ├── pdf/
│   └── html/
└── processed/
    └── chunks/
```

---

## Quality Assurance

### Human Validation Checklist
- [ ] Question is clear and unambiguous
- [ ] Answer is factually correct
- [ ] Sources are properly cited
- [ ] Query type and difficulty are accurate
- [ ] No PII or sensitive information

### Automated Validation
```python
def validate_golden_pair(qa: dict) -> bool:
    checks = [
        len(qa['query']) > 10,  # Minimum query length
        len(qa['ground_truth_answer']) > 20,  # Minimum answer length
        len(qa['ground_truth_contexts']) > 0,  # At least one source
        qa['query'].endswith('?'),  # Proper question format
        qa['metadata']['query_type'] in VALID_TYPES,
    ]
    return all(checks)
```

---

## Implementation Code

See:
- `src/dataset/parser.py` - Document parsing logic
- `src/dataset/golden_set.py` - Golden set management
- `src/dataset/synthetic_generator.py` - LLM-based QA generation

---

## Next Steps

Proceed to **Step 4: Data Ingestion & Chunking Experimentation Framework** to implement the document processing pipeline.
