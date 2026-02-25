#!/usr/bin/env python
"""CLI script for document ingestion."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunker import ChunkingStrategy
from src.retrieval.pinecone_client import PineconeClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run document ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--input",
        type=Path,
        default="data/raw",
        help="Input directory with raw documents"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default="data/cache",
        help="Cache directory for checkpoints"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["recursive", "semantic", "markdown"],
        default="recursive",
        help="Chunking strategy"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index chunks to Pinecone"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Pinecone namespace"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/processed/chunks.jsonl",
        help="Output file for chunks"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Output: {args.output}")

    # Initialize pipeline
    strategy = ChunkingStrategy(args.strategy)
    pipeline = IngestionPipeline(
        data_dir=args.input,
        cache_dir=args.cache,
        chunking_strategy=strategy
    )

    # Run ingestion
    try:
        chunks = pipeline.run(output_file=args.output)

        if not chunks:
            logger.warning("No chunks produced")
            return

        # Index to Pinecone if requested
        if args.index:
            logger.info(f"Indexing {len(chunks)} chunks to Pinecone...")
            client = PineconeClient()
            client.upsert_chunks(chunks, namespace=args.namespace)
            logger.info("Indexing complete")

        logger.info("=" * 60)
        logger.info(f"✓ Ingestion complete: {len(chunks)} chunks processed")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
