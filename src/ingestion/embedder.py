"""Embedding generation for dense and sparse vectors."""

from typing import List, Dict
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder

from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class HybridEmbedder:
    """Generate both dense and sparse embeddings for hybrid search."""

    def __init__(self):
        """Initialize embedders."""
        # Dense embeddings (OpenAI)
        self.dense_embedder = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )

        # Sparse embeddings (BM25)
        self.sparse_embedder = BM25Encoder()
        self.sparse_fitted = False

    def fit_sparse(self, texts: List[str]):
        """
        Fit BM25 encoder on corpus.

        Args:
            texts: List of all text chunks
        """
        logger.info(f"Fitting BM25 encoder on {len(texts)} texts")
        self.sparse_embedder.fit(texts)
        self.sparse_fitted = True

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Chunks with embeddings added
        """
        texts = [chunk['text'] for chunk in chunks]

        # Generate dense embeddings
        logger.info(f"Generating dense embeddings for {len(texts)} chunks")
        try:
            dense_vectors = self.dense_embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating dense embeddings: {e}")
            raise

        # Generate sparse embeddings
        if not self.sparse_fitted:
            self.fit_sparse(texts)

        logger.info(f"Generating sparse embeddings for {len(texts)} chunks")
        try:
            sparse_vectors = self.sparse_embedder.encode_documents(texts)
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            raise

        # Attach embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['dense_embedding'] = dense_vectors[i]
            chunk['sparse_embedding'] = {
                'indices': sparse_vectors[i]['indices'].tolist() if hasattr(sparse_vectors[i]['indices'], 'tolist') else sparse_vectors[i]['indices'],
                'values': sparse_vectors[i]['values'].tolist() if hasattr(sparse_vectors[i]['values'], 'tolist') else sparse_vectors[i]['values']
            }

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return chunks

    def embed_query(self, query: str) -> tuple:
        """
        Generate embeddings for a query.

        Args:
            query: Query text

        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        # Dense embedding
        dense_vector = self.dense_embedder.embed_query(query)

        # Sparse embedding
        if not self.sparse_fitted:
            logger.warning("BM25 encoder not fitted, returning empty sparse vector")
            sparse_vector = {'indices': [], 'values': []}
        else:
            sparse_result = self.sparse_embedder.encode_queries([query])[0]
            sparse_vector = {
                'indices': sparse_result['indices'].tolist() if hasattr(sparse_result['indices'], 'tolist') else sparse_result['indices'],
                'values': sparse_result['values'].tolist() if hasattr(sparse_result['values'], 'tolist') else sparse_result['values']
            }

        return dense_vector, sparse_vector


def compute_embedding_stats(embeddings: List[List[float]]) -> Dict:
    """
    Compute statistics on embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Dictionary with statistics
    """
    embeddings_array = np.array(embeddings)

    return {
        'count': len(embeddings),
        'dimensions': embeddings_array.shape[1] if len(embeddings) > 0 else 0,
        'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
    }
