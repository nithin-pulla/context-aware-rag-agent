#!/usr/bin/env python
"""CLI script for evaluating the RAG system using RAGAs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ragas_evaluator import RAGEvaluator
from src.dataset.golden_set import GoldenSetManager
from src.query.engine import RAGQueryEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Evaluate RAG system on golden set."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system with RAGAs")
    parser.add_argument(
        "--golden-set",
        type=Path,
        default="data/golden_set",
        help="Golden set directory"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="current",
        help="Golden set version"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="reports/evaluation_latest.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--threshold-gate",
        action="store_true",
        help="Fail if thresholds not met (for CI/CD)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline results for regression detection"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RAG System Evaluation (RAGAs)")
    logger.info("=" * 60)

    try:
        # Load golden set
        logger.info(f"Loading golden set from {args.golden_set}...")
        manager = GoldenSetManager(args.golden_set)
        golden_set = manager.load_golden_set(version=args.version)

        if not golden_set:
            logger.error("No golden set found")
            sys.exit(1)

        logger.info(f"Loaded {len(golden_set)} QA pairs")

        # Initialize components
        logger.info("Initializing query engine...")
        query_engine = RAGQueryEngine()

        logger.info("Initializing evaluator...")
        evaluator = RAGEvaluator(query_engine)

        # Run evaluation
        logger.info("Running evaluation...")
        metrics = evaluator.evaluate_golden_set(golden_set)

        # Save results
        evaluator.save_results(metrics, args.output)

        # Check thresholds
        passed = evaluator.check_thresholds(metrics)

        # Check for regressions
        if args.baseline:
            has_regression = evaluator.detect_regression(
                metrics,
                args.baseline
            )
            if has_regression:
                logger.error("Regression detected compared to baseline")
                if args.threshold_gate:
                    sys.exit(1)

        # Exit with error if threshold gate enabled and checks failed
        if args.threshold_gate and not passed:
            logger.error("Threshold checks failed")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("✓ Evaluation complete")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
