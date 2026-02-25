"""End-to-end ingestion pipeline."""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.dataset.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker, ChunkingStrategy
from src.ingestion.embedder import HybridEmbedder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class IngestionCheckpoint:
    """Manage checkpointing for incremental processing."""

    def __init__(self, checkpoint_file: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = checkpoint_file
        self.processed = self._load()

    def _load(self) -> set:
        """Load processed document hashes."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()

    def is_processed(self, doc_hash: str) -> bool:
        """Check if document already processed."""
        return doc_hash in self.processed

    def mark_processed(self, doc_hash: str):
        """Mark document as processed."""
        self.processed.add(doc_hash)
        self._save()

    def _save(self):
        """Save checkpoint to disk."""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed), f, indent=2)


class DeadLetterQueue:
    """Store failed documents for manual review."""

    def __init__(self, dlq_file: Path):
        """
        Initialize dead letter queue.

        Args:
            dlq_file: Path to DLQ file
        """
        self.dlq_file = dlq_file
        self.dlq_file.parent.mkdir(parents=True, exist_ok=True)

    def add(self, doc: Dict, error: Exception):
        """
        Add failed document to queue.

        Args:
            doc: Document dictionary
            error: Exception that caused failure
        """
        with jsonlines.open(self.dlq_file, mode='a') as writer:
            writer.write({
                'source': doc.get('source', 'unknown'),
                'error': str(error),
                'error_type': type(error).__name__,
                'timestamp': datetime.now().isoformat(),
                'metadata': doc.get('metadata', {})
            })
        logger.error(f"Added document to DLQ: {doc.get('source')} - {error}")


class IngestionPipeline:
    """Orchestrate document ingestion pipeline."""

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        skip_existing: bool = True
    ):
        """
        Initialize ingestion pipeline.

        Args:
            data_dir: Directory containing raw documents
            cache_dir: Directory for caching and checkpoints
            chunking_strategy: Strategy for chunking documents
            skip_existing: Skip already processed documents
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.parser = DocumentParser()
        self.chunker = DocumentChunker(strategy=chunking_strategy)
        self.embedder = HybridEmbedder()

        self.checkpoint = IngestionCheckpoint(cache_dir / "checkpoint.json") if skip_existing else None
        self.dlq = DeadLetterQueue(cache_dir / "dead_letter_queue.jsonl")

    def load_documents(self) -> List[Dict]:
        """
        Load and parse all documents from data directory.

        Returns:
            List of parsed documents
        """
        logger.info(f"Loading documents from {self.data_dir}")
        documents = []

        for file_path in self.data_dir.rglob('*'):
            if not file_path.is_file():
                continue

            try:
                parsed = self.parser.parse(file_path)
                if not parsed:
                    continue

                doc_hash = parsed['metadata']['content_hash']

                # Skip if already processed
                if self.checkpoint and self.checkpoint.is_processed(doc_hash):
                    logger.info(f"Skipping already processed: {file_path}")
                    continue

                documents.append(parsed)

            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                self.dlq.add({'source': str(file_path)}, e)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process documents through chunking and embedding.

        Args:
            documents: List of parsed documents

        Returns:
            List of chunks with embeddings
        """
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self.chunker.chunk_document(doc)
                chunks.extend(doc_chunks)

                # Mark as processed
                if self.checkpoint:
                    self.checkpoint.mark_processed(doc['metadata']['content_hash'])

            except Exception as e:
                logger.error(f"Error chunking document {doc['source']}: {e}")
                self.dlq.add(doc, e)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Generate embeddings
        if chunks:
            logger.info("Generating embeddings...")
            try:
                chunks = self.embedder.embed_chunks(chunks)
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        return chunks

    def save_chunks(self, chunks: List[Dict], output_file: Path):
        """
        Save processed chunks to file.

        Args:
            chunks: List of chunks with embeddings
            output_file: Output file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(chunks)} chunks to {output_file}")
        with jsonlines.open(output_file, mode='w') as writer:
            for chunk in chunks:
                writer.write(chunk)

        logger.info(f"Successfully saved chunks to {output_file}")

    def run(self, output_file: Optional[Path] = None) -> List[Dict]:
        """
        Run the complete ingestion pipeline.

        Args:
            output_file: Optional file to save chunks

        Returns:
            List of processed chunks
        """
        logger.info("Starting ingestion pipeline")

        # Load documents
        documents = self.load_documents()

        if not documents:
            logger.warning("No documents to process")
            return []

        # Process documents
        chunks = self.process_documents(documents)

        # Save chunks if output file specified
        if output_file:
            self.save_chunks(chunks, output_file)

        logger.info(f"Ingestion pipeline completed. Processed {len(chunks)} chunks.")
        return chunks
