"""Hybrid search combining dense and sparse retrieval."""

from typing import List, Dict, Optional
from src.ingestion.embedder import HybridEmbedder
from src.retrieval.pinecone_client import PineconeClient
from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class HybridSearchEngine:
    """Hybrid search engine with alpha-tunable retrieval."""

    def __init__(
        self,
        pinecone_client: Optional[PineconeClient] = None,
        embedder: Optional[HybridEmbedder] = None,
        alpha: float = None,
        top_k: int = None
    ):
        """
        Initialize hybrid search engine.

        Args:
            pinecone_client: Pinecone client instance
            embedder: Hybrid embedder instance
            alpha: Weight for dense vs sparse (0=sparse, 1=dense)
            top_k: Number of results to retrieve
        """
        self.pinecone_client = pinecone_client or PineconeClient()
        self.embedder = embedder or HybridEmbedder()
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.top_k = top_k or settings.top_k

        # Ensure connection
        self.pinecone_client.connect()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        namespace: str = "default",
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results (override default)
            alpha: Hybrid weight (override default)
            namespace: Pinecone namespace
            filter: Metadata filter

        Returns:
            List of search results
        """
        k = top_k or self.top_k
        search_alpha = alpha if alpha is not None else self.alpha

        logger.info(f"Hybrid search: query='{query[:50]}...', top_k={k}, alpha={search_alpha}")

        try:
            # Generate embeddings
            dense_vector, sparse_vector = self.embedder.embed_query(query)

            # Query Pinecone
            results = self.pinecone_client.query(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )

            # Format results
            formatted_results = []
            for match in results:
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'text': match['metadata'].get('text', ''),
                    'source': match['metadata'].get('source', ''),
                    'chunk_index': match['metadata'].get('chunk_index', 0),
                    'metadata': match['metadata']
                })

            logger.info(f"Retrieved {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise

    def search_with_context_window(
        self,
        query: str,
        top_k: int = 3,
        context_window: int = 1
    ) -> List[Dict]:
        """
        Search and include surrounding chunks for context.

        Args:
            query: Search query
            top_k: Number of primary results
            context_window: Number of chunks before/after to include

        Returns:
            Results with expanded context
        """
        # Get primary results
        results = self.search(query, top_k=top_k)

        # TODO: Implement context window expansion
        # Would require storing chunk ordering metadata and fetching adjacent chunks

        return results

    def tune_alpha(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        alpha_range: List[float] = None
    ) -> float:
        """
        Find optimal alpha using grid search on golden set.

        Args:
            queries: List of test queries
            ground_truth: List of expected doc IDs for each query
            alpha_range: List of alpha values to test

        Returns:
            Optimal alpha value
        """
        if alpha_range is None:
            alpha_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        best_alpha = 0.5
        best_score = 0.0

        logger.info(f"Tuning alpha over {len(alpha_range)} values")

        for alpha in alpha_range:
            # Evaluate with this alpha
            precision_scores = []

            for query, truth in zip(queries, ground_truth):
                results = self.search(query, top_k=len(truth), alpha=alpha)
                retrieved_ids = [r['id'] for r in results]

                # Compute precision@k
                relevant = len(set(retrieved_ids[:len(truth)]) & set(truth))
                precision = relevant / len(truth) if truth else 0
                precision_scores.append(precision)

            avg_precision = sum(precision_scores) / len(precision_scores)
            logger.info(f"Alpha={alpha:.1f}: Precision={avg_precision:.3f}")

            if avg_precision > best_score:
                best_score = avg_precision
                best_alpha = alpha

        logger.info(f"Optimal alpha: {best_alpha} (Precision={best_score:.3f})")
        return best_alpha


def compute_reciprocal_rank_fusion(
    rankings: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    Args:
        rankings: List of result lists from different retrievers
        k: RRF parameter (default: 60)

    Returns:
        Fused ranking
    """
    scores = {}
    doc_map = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc['id']
            score = 1.0 / (k + rank + 1)

            if doc_id in scores:
                scores[doc_id] += score
            else:
                scores[doc_id] = score
                doc_map[doc_id] = doc

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_map[doc_id] for doc_id in sorted_ids]
