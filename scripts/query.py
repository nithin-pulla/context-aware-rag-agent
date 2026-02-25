#!/usr/bin/env python
"""CLI script for querying the RAG system."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.engine import RAGQueryEngine, format_response_for_display
from src.observability.cost_tracker import cost_tracker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Query the RAG system."""
    parser = argparse.ArgumentParser(description="Query RAG system")
    parser.add_argument(
        "query",
        type=str,
        help="Query text"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--rerank",
        type=int,
        default=3,
        help="Number of documents to keep after re-ranking"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable re-ranking"
    )
    parser.add_argument(
        "--track-cost",
        action="store_true",
        help="Track and display query cost"
    )

    args = parser.parse_args()

    logger.info("Initializing RAG Query Engine...")

    try:
        # Initialize engine
        engine = RAGQueryEngine(use_reranking=not args.no_rerank)

        # Process query
        logger.info(f"Processing query: '{args.query}'")
        response = engine.query(
            query=args.query,
            top_k=args.top_k,
            rerank_top_n=args.rerank
        )

        # Display response
        print("\n" + "=" * 80)
        print(format_response_for_display(response))
        print("=" * 80)

        # Track cost if requested
        if args.track_cost and response.get('answer'):
            cost_breakdown = cost_tracker.track_query_cost(
                query=args.query,
                contexts=response.get('sources', []),
                response=response['answer']
            )
            print("\nCost Breakdown:")
            print("-" * 80)
            print(f"Embedding: ${cost_breakdown['embedding_cost']:.4f}")
            print(f"LLM: ${cost_breakdown['llm_cost']:.4f}")
            print(f"Pinecone: ${cost_breakdown['pinecone_cost']:.6f}")
            print(f"Total: ${cost_breakdown['total_cost']:.4f}")
            print("=" * 80)

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
