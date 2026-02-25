"""Main query engine orchestrating the RAG pipeline."""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.query.prompts import RAG_PROMPT_TEMPLATE
from src.query.validator import QueryValidator
from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class RAGQueryEngine:
    """End-to-end RAG query engine."""

    def __init__(
        self,
        search_engine: Optional[HybridSearchEngine] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        use_reranking: bool = True
    ):
        """
        Initialize query engine.

        Args:
            search_engine: Hybrid search engine instance
            reranker: Cross-encoder reranker instance
            use_reranking: Whether to use re-ranking
        """
        self.search_engine = search_engine or HybridSearchEngine()
        self.reranker = reranker if use_reranking else None
        self.validator = QueryValidator()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
            request_timeout=settings.timeout_seconds
        )

        # Build RAG chain
        self.chain = RAG_PROMPT_TEMPLATE | self.llm | StrOutputParser()

        logger.info("RAG Query Engine initialized")

    def query(
        self,
        query: str,
        top_k: int = None,
        rerank_top_n: int = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Process a user query through the RAG pipeline.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            rerank_top_n: Number of documents after re-ranking
            include_sources: Include source documents in response

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{query[:100]}...'")

        # Validate and sanitize query
        is_valid, sanitized_query, error = self.validator.process(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {error}")
            return {
                "answer": None,
                "error": error,
                "sources": [],
                "metadata": {"status": "validation_failed"}
            }

        try:
            # Retrieve relevant documents
            top_k = top_k or settings.top_k
            logger.info(f"Retrieving top-{top_k} documents")
            results = self.search_engine.search(
                query=sanitized_query,
                top_k=top_k
            )

            if not results:
                logger.warning("No results found")
                return {
                    "answer": "I couldn't find any relevant documentation to answer your question.",
                    "sources": [],
                    "metadata": {"status": "no_results"}
                }

            # Re-rank if enabled
            if self.reranker:
                rerank_n = rerank_top_n or settings.rerank_top_n
                logger.info(f"Re-ranking to top-{rerank_n}")
                results = self.reranker.rerank(
                    query=sanitized_query,
                    results=results,
                    top_n=rerank_n
                )

            # Format context
            context = self._format_context(results)

            # Generate answer
            logger.info("Generating answer with LLM")
            answer = self.chain.invoke({
                "query": sanitized_query,
                "context": context
            })

            # Format response
            response = {
                "answer": answer,
                "sources": results if include_sources else [],
                "metadata": {
                    "status": "success",
                    "num_sources": len(results),
                    "retrieval_scores": [r.get('score', 0) for r in results],
                }
            }

            if self.reranker and results:
                response["metadata"]["rerank_scores"] = [
                    r.get('rerank_score', 0) for r in results
                ]

            logger.info("Query processed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": None,
                "error": f"An error occurred: {str(e)}",
                "sources": [],
                "metadata": {"status": "error"}
            }

    def _format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            results: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            source = result.get('source', 'Unknown')
            text = result.get('text', '')

            context_parts.append(
                f"[Document {i}] Source: {source}\n{text}\n"
            )

        return "\n---\n\n".join(context_parts)

    def batch_query(self, queries: List[str]) -> List[Dict]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of queries

        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        responses = []

        for query in queries:
            response = self.query(query)
            responses.append(response)

        return responses


def format_response_for_display(response: Dict) -> str:
    """
    Format response for console/CLI display.

    Args:
        response: Response dictionary from query engine

    Returns:
        Formatted string for display
    """
    if response.get('error'):
        return f"Error: {response['error']}"

    output = []

    # Answer
    output.append("Answer:")
    output.append("-" * 80)
    output.append(response['answer'])
    output.append("")

    # Sources
    if response.get('sources'):
        output.append("Sources:")
        output.append("-" * 80)
        for i, source in enumerate(response['sources'], 1):
            output.append(f"[{i}] {source.get('source', 'Unknown')}")
            output.append(f"    Score: {source.get('rerank_score', source.get('score', 0)):.3f}")
            output.append(f"    Preview: {source.get('text', '')[:150]}...")
            output.append("")

    return "\n".join(output)
