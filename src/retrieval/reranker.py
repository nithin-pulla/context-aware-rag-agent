"""Cross-encoder re-ranking for improved precision."""

from typing import List, Dict
from sentence_transformers import CrossEncoder

from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class CrossEncoderReranker:
    """Re-rank results using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_n: int = None
    ) -> List[Dict]:
        """
        Re-rank search results using cross-encoder.

        Args:
            query: Original query
            results: List of search results
            top_n: Number of top results to return

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        top_n = top_n or settings.rerank_top_n

        logger.info(f"Re-ranking {len(results)} results (returning top {top_n})")

        try:
            # Prepare query-document pairs
            pairs = [[query, result['text']] for result in results]

            # Compute cross-encoder scores
            scores = self.model.predict(pairs)

            # Attach scores and sort
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])
                result['original_rank'] = i + 1

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

            logger.info(f"Re-ranking complete. Top score: {reranked[0]['rerank_score']:.3f}")

            return reranked[:top_n]

        except Exception as e:
            logger.error(f"Error re-ranking results: {e}")
            # Fallback to original ranking
            return results[:top_n]

    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Dict]],
        top_n: int = None
    ) -> List[List[Dict]]:
        """
        Re-rank results for multiple queries in batch.

        Args:
            queries: List of queries
            results_list: List of result lists (one per query)
            top_n: Number of top results per query

        Returns:
            List of re-ranked result lists
        """
        reranked_list = []

        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_n)
            reranked_list.append(reranked)

        return reranked_list


class MMRReranker:
    """Maximal Marginal Relevance re-ranker for diversity."""

    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR re-ranker.

        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        query_embedding: List[float],
        results: List[Dict],
        top_n: int = 3
    ) -> List[Dict]:
        """
        Re-rank using MMR for diversity.

        Args:
            query_embedding: Query embedding vector
            results: Search results with embeddings
            top_n: Number of results to return

        Returns:
            MMR re-ranked results
        """
        import numpy as np

        if not results:
            return []

        # Extract embeddings
        doc_embeddings = [r.get('dense_embedding', []) for r in results]
        if not all(doc_embeddings):
            logger.warning("Not all results have embeddings, falling back to original order")
            return results[:top_n]

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)

        # Compute similarity to query
        query_sim = np.dot(doc_vecs, query_vec)

        selected = []
        remaining = list(range(len(results)))

        for _ in range(min(top_n, len(results))):
            mmr_scores = []

            for i in remaining:
                # Relevance score
                relevance = query_sim[i]

                # Diversity score (max similarity to already selected)
                if selected:
                    selected_vecs = doc_vecs[selected]
                    diversity = -np.max(np.dot(selected_vecs, doc_vecs[i]))
                else:
                    diversity = 0

                # MMR score
                mmr = self.lambda_param * relevance + (1 - self.lambda_param) * diversity
                mmr_scores.append(mmr)

            # Select best MMR score
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [results[i] for i in selected]
