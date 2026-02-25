"""Document chunking strategies."""

from typing import List, Dict
from enum import Enum
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"


class DocumentChunker:
    """Chunk documents using various strategies."""

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document chunker.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.strategy = strategy
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = self._create_splitter()

    def _create_splitter(self):
        """Create text splitter based on strategy."""
        if self.strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )

        elif self.strategy == ChunkingStrategy.SEMANTIC:
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=70,
            )

        elif self.strategy == ChunkingStrategy.MARKDOWN:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )

        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a single document.

        Args:
            document: Parsed document dictionary

        Returns:
            List of chunk dictionaries
        """
        text = document['full_text']
        source = document['source']
        doc_hash = document['metadata']['content_hash']

        try:
            # Split text
            if self.strategy == ChunkingStrategy.MARKDOWN and document['metadata']['format'] == 'markdown':
                chunk_texts = self.splitter.split_text(text)
            else:
                chunk_texts = self.splitter.split_text(text)

            # Create chunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                # Skip very small chunks
                if len(chunk_text.strip()) < 50:
                    continue

                chunk = {
                    'chunk_id': f"{doc_hash}:chunk_{i}",
                    'text': chunk_text.strip(),
                    'source': source,
                    'chunk_index': i,
                    'total_chunks': len(chunk_texts),
                    'metadata': {
                        **document['metadata'],
                        'chunking_strategy': self.strategy.value,
                        'chunk_size_setting': self.chunk_size,
                    }
                }
                chunks.append(chunk)

            logger.info(f"Chunked {source} into {len(chunks)} chunks using {self.strategy.value}")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document {source}: {e}")
            raise

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of parsed documents

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters for English
    return len(text) // 4
