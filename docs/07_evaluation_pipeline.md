# Step 7: Evaluation Pipeline (RAGAs Integration)

## Overview

The evaluation pipeline uses RAGAs (Retrieval-Augmented Generation Assessment) to measure system quality across multiple dimensions. This provides objective, automated assessment of the RAG system.

## RAGAs Metrics

### 1. Faithfulness

**Definition**: Measures whether the generated answer is grounded in the retrieved context.

**Formula**: Fraction of claims in the answer that can be verified from the context.

**Scoring**: 0.0 - 1.0 (higher is better)

**Example**:
- Context: "The API rate limit is 100 requests per minute."
- Answer: "The API has a rate limit of 100 requests per minute."
- Faithfulness: 1.0 (fully grounded)

- Answer: "The API has a rate limit of 1000 requests per minute."
- Faithfulness: 0.0 (hallucinated number)

**Threshold**: ≥ 0.70

---

### 2. Answer Relevance

**Definition**: Measures how relevant the answer is to the question.

**Formula**: Semantic similarity between question and answer.

**Scoring**: 0.0 - 1.0 (higher is better)

**Example**:
- Question: "How do I authenticate?"
- Answer: "Use OAuth 2.0 with the Authorization header."
- Relevance: 0.95 (highly relevant)

- Answer: "The API supports multiple endpoints."
- Relevance: 0.20 (not relevant)

**Threshold**: ≥ 0.75

---

### 3. Context Precision

**Definition**: Measures what fraction of retrieved contexts are relevant to the question.

**Formula**: Fraction of top-k retrieved docs that are actually relevant.

**Scoring**: 0.0 - 1.0 (higher is better)

**Example**:
- Retrieved 5 docs, 4 are relevant
- Context Precision: 0.80

**Threshold**: ≥ 0.80

---

### 4. Context Recall

**Definition**: Measures what fraction of necessary information was retrieved.

**Formula**: Fraction of ground truth contexts that were retrieved.

**Scoring**: 0.0 - 1.0 (higher is better)

**Example**:
- Ground truth requires 3 specific contexts
- System retrieved 2 of them
- Context Recall: 0.67

**Threshold**: ≥ 0.70

---

## Evaluation Pipeline Architecture

```
┌─────────────────┐
│  Golden Set     │ (QA pairs with ground truth)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Executor  │ (Run each query through system)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RAGAs Evaluator │ (Compute metrics)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Generator│ (JSON, HTML, CSV)
└─────────────────┘
```

---

## Implementation

### Basic Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    'question': [...],  # User queries
    'answer': [...],  # Generated answers
    'contexts': [...],  # Retrieved contexts (list of lists)
    'ground_truth': [...]  # Ground truth answers
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(result)
```

### Custom Evaluation Runner

```python
class RAGEvaluator:
    """Evaluate RAG system using RAGAs."""

    def __init__(self, query_engine, golden_set):
        self.query_engine = query_engine
        self.golden_set = golden_set

    def evaluate(self):
        """Run evaluation on golden set."""
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        # Run queries
        for qa_pair in self.golden_set:
            response = self.query_engine.query(qa_pair.query)

            questions.append(qa_pair.query)
            answers.append(response['answer'])
            contexts.append([s['text'] for s in response['sources']])
            ground_truths.append(qa_pair.ground_truth_answer)

        # Create dataset
        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        })

        # Evaluate
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        return results
```

---

## Threshold Gating for CI/CD

### Pipeline Gate

```python
def gate_evaluation(results: dict, thresholds: dict) -> bool:
    """
    Check if evaluation meets quality thresholds.

    Args:
        results: RAGAs evaluation results
        thresholds: Dictionary of metric thresholds

    Returns:
        True if all thresholds met, False otherwise
    """
    checks = {
        'faithfulness': results['faithfulness'] >= thresholds['faithfulness'],
        'answer_relevancy': results['answer_relevancy'] >= thresholds['answer_relevancy'],
        'context_precision': results['context_precision'] >= thresholds['context_precision'],
        'context_recall': results['context_recall'] >= thresholds['context_recall'],
    }

    passed = all(checks.values())

    logger.info(f"Evaluation gate: {'PASSED' if passed else 'FAILED'}")
    for metric, check in checks.items():
        status = '✓' if check else '✗'
        logger.info(f"  {status} {metric}: {results[metric]:.3f}")

    return passed
```

### CI/CD Integration

```yaml
# .github/workflows/evaluate.yml
name: RAG Evaluation

on:
  pull_request:
  push:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
        run: |
          python -m src.evaluation.evaluate --threshold-gate

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-report
          path: reports/evaluation_*.json
```

---

## Regression Detection

### Historical Comparison

```python
def detect_regression(current_results: dict, baseline_file: str) -> bool:
    """
    Compare current results against baseline.

    Args:
        current_results: Current evaluation results
        baseline_file: Path to baseline results JSON

    Returns:
        True if regression detected
    """
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    REGRESSION_THRESHOLD = 0.05  # 5% drop is a regression

    regressions = []
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        current = current_results[metric]
        baseline_val = baseline[metric]
        delta = current - baseline_val

        if delta < -REGRESSION_THRESHOLD:
            regressions.append({
                'metric': metric,
                'baseline': baseline_val,
                'current': current,
                'delta': delta
            })

    if regressions:
        logger.error("REGRESSIONS DETECTED:")
        for r in regressions:
            logger.error(f"  {r['metric']}: {r['baseline']:.3f} → {r['current']:.3f} ({r['delta']:.3f})")
        return True

    return False
```

---

## Report Generation

### JSON Report

```python
def generate_json_report(results: dict, output_file: str):
    """Generate JSON evaluation report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'faithfulness': float(results['faithfulness']),
            'answer_relevancy': float(results['answer_relevancy']),
            'context_precision': float(results['context_precision']),
            'context_recall': float(results['context_recall']),
        },
        'summary': {
            'average_score': float(np.mean(list(results.values()))),
            'passed_thresholds': gate_evaluation(results, THRESHOLDS),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
```

### HTML Dashboard

```html
<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>RAG System Evaluation</h1>

    <div id="metrics-chart"></div>

    <script>
        var data = [{
            type: 'bar',
            x: ['Faithfulness', 'Relevancy', 'Precision', 'Recall'],
            y: [0.85, 0.82, 0.88, 0.75],
            marker: {color: ['#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c']}
        }];

        var layout = {
            title: 'RAGAs Metrics',
            yaxis: {range: [0, 1], title: 'Score'},
            shapes: [{
                type: 'line',
                x0: -0.5,
                x1: 3.5,
                y0: 0.7,
                y1: 0.7,
                line: {color: 'red', dash: 'dash', width: 2}
            }]
        };

        Plotly.newPlot('metrics-chart', data, layout);
    </script>
</body>
</html>
```

---

## Batch Evaluation Scheduling

### Nightly Evaluation

```python
def scheduled_evaluation():
    """Run nightly evaluation on golden set."""
    logger.info("Starting scheduled evaluation")

    # Load golden set
    golden_set = load_golden_set()

    # Run evaluation
    evaluator = RAGEvaluator(query_engine, golden_set)
    results = evaluator.evaluate()

    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generate_json_report(results, f"reports/eval_{timestamp}.json")
    generate_html_report(results, f"reports/eval_{timestamp}.html")

    # Check for regressions
    if detect_regression(results, "reports/baseline.json"):
        send_alert("RAG evaluation regression detected!")

    logger.info("Scheduled evaluation complete")
```

---

## Implementation Files

- `src/evaluation/__init__.py` - Evaluation module
- `src/evaluation/ragas_evaluator.py` - RAGAs integration
- `src/evaluation/report_generator.py` - Report generation
- `src/evaluation/regression_detector.py` - Regression detection

---

## Next Steps

Proceed to **Step 8: Metrics, Observability & Cost Engineering** to implement production monitoring and cost tracking.
