"""RAGAs evaluation implementation."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

from src.dataset.golden_set import GoldenSetManager, GoldenQAPair
from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class RAGEvaluator:
    """Evaluate RAG system using RAGAs metrics."""

    THRESHOLDS = {
        'faithfulness': settings.faithfulness_threshold,
        'answer_relevancy': settings.relevance_threshold,
        'context_precision': settings.precision_threshold,
        'context_recall': settings.recall_threshold,
    }

    def __init__(self, query_engine):
        """
        Initialize evaluator.

        Args:
            query_engine: RAG query engine instance
        """
        self.query_engine = query_engine

    def evaluate_golden_set(
        self,
        golden_set: List[GoldenQAPair],
        batch_size: int = None
    ) -> Dict:
        """
        Evaluate on golden QA pairs.

        Args:
            golden_set: List of golden QA pairs
            batch_size: Batch size for processing

        Returns:
            Dictionary with RAGAs metrics
        """
        batch_size = batch_size or settings.batch_size

        logger.info(f"Evaluating on {len(golden_set)} golden QA pairs")

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        # Process in batches
        for i in range(0, len(golden_set), batch_size):
            batch = golden_set[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(golden_set)-1)//batch_size + 1}")

            for qa_pair in batch:
                try:
                    # Run query
                    response = self.query_engine.query(qa_pair.query)

                    if response.get('error'):
                        logger.warning(f"Query failed for {qa_pair.id}: {response['error']}")
                        continue

                    # Collect data
                    questions.append(qa_pair.query)
                    answers.append(response['answer'])
                    contexts.append([s['text'] for s in response['sources']])
                    ground_truths.append(qa_pair.ground_truth_answer)

                except Exception as e:
                    logger.error(f"Error processing {qa_pair.id}: {e}")

        if not questions:
            logger.error("No successful queries to evaluate")
            return {}

        # Create dataset for RAGAs
        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        })

        # Run RAGAs evaluation
        logger.info("Computing RAGAs metrics...")
        try:
            results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ]
            )

            # Convert to dict
            metrics = {
                'faithfulness': float(results['faithfulness']),
                'answer_relevancy': float(results['answer_relevancy']),
                'context_precision': float(results['context_precision']),
                'context_recall': float(results['context_recall']),
            }

            logger.info("Evaluation complete")
            self._log_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error computing RAGAs metrics: {e}")
            raise

    def _log_metrics(self, metrics: Dict):
        """Log metrics with threshold comparison."""
        logger.info("=" * 60)
        logger.info("RAGAs Evaluation Results")
        logger.info("=" * 60)

        for metric_name, value in metrics.items():
            threshold = self.THRESHOLDS.get(metric_name, 0)
            status = "✓" if value >= threshold else "✗"
            logger.info(f"{status} {metric_name:20s}: {value:.3f} (threshold: {threshold:.2f})")

        avg_score = sum(metrics.values()) / len(metrics)
        logger.info("-" * 60)
        logger.info(f"  Average Score: {avg_score:.3f}")
        logger.info("=" * 60)

    def check_thresholds(self, metrics: Dict) -> bool:
        """
        Check if metrics meet thresholds.

        Args:
            metrics: Dictionary of metric values

        Returns:
            True if all thresholds met
        """
        checks = {
            metric: value >= self.THRESHOLDS[metric]
            for metric, value in metrics.items()
            if metric in self.THRESHOLDS
        }

        passed = all(checks.values())

        if passed:
            logger.info("✓ All threshold checks PASSED")
        else:
            logger.error("✗ Threshold checks FAILED")
            for metric, check in checks.items():
                if not check:
                    logger.error(f"  ✗ {metric}: {metrics[metric]:.3f} < {self.THRESHOLDS[metric]:.2f}")

        return passed

    def detect_regression(
        self,
        current_metrics: Dict,
        baseline_file: Path,
        threshold: float = 0.05
    ) -> bool:
        """
        Detect regression compared to baseline.

        Args:
            current_metrics: Current evaluation metrics
            baseline_file: Path to baseline metrics JSON
            threshold: Regression threshold (e.g., 0.05 = 5% drop)

        Returns:
            True if regression detected
        """
        if not baseline_file.exists():
            logger.warning(f"Baseline file not found: {baseline_file}")
            return False

        with open(baseline_file, 'r') as f:
            baseline = json.load(f)

        regressions = []
        for metric in current_metrics.keys():
            if metric not in baseline:
                continue

            current = current_metrics[metric]
            baseline_val = baseline[metric]
            delta = current - baseline_val

            if delta < -threshold:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline_val,
                    'current': current,
                    'delta': delta,
                    'percent_change': (delta / baseline_val) * 100
                })

        if regressions:
            logger.error("=" * 60)
            logger.error("REGRESSIONS DETECTED")
            logger.error("=" * 60)
            for r in regressions:
                logger.error(
                    f"✗ {r['metric']}: {r['baseline']:.3f} → {r['current']:.3f} "
                    f"({r['delta']:.3f}, {r['percent_change']:.1f}%)"
                )
            logger.error("=" * 60)
            return True

        logger.info("✓ No regressions detected")
        return False

    def save_results(
        self,
        metrics: Dict,
        output_file: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Save evaluation results to file.

        Args:
            metrics: Evaluation metrics
            output_file: Output file path
            metadata: Optional metadata to include
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'thresholds': self.THRESHOLDS,
            'passed': self.check_thresholds(metrics),
            'average_score': sum(metrics.values()) / len(metrics)
        }

        if metadata:
            results['metadata'] = metadata

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")
