"""Cost tracking for RAG system operations."""

import tiktoken
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CostTracker:
    """Track API costs for RAG operations."""

    # Pricing (as of 2024, subject to change)
    PRICING = {
        'embedding': 0.00002,  # per 1K tokens (text-embedding-3-small)
        'llm_input': 0.01,     # per 1K tokens (GPT-4 Turbo)
        'llm_output': 0.03,    # per 1K tokens (GPT-4 Turbo)
        'pinecone_query': 0.0000003,  # per query (Serverless)
    }

    def __init__(self):
        """Initialize cost tracker."""
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.total_cost = 0.0
        self.query_count = 0

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.encoding.encode(text))

    def compute_embedding_cost(self, tokens: int) -> float:
        """
        Compute embedding cost.

        Args:
            tokens: Number of tokens

        Returns:
            Cost in dollars
        """
        return (tokens / 1000) * self.PRICING['embedding']

    def compute_llm_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Compute LLM generation cost.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in dollars
        """
        input_cost = (input_tokens / 1000) * self.PRICING['llm_input']
        output_cost = (output_tokens / 1000) * self.PRICING['llm_output']
        return input_cost + output_cost

    def compute_pinecone_cost(self, queries: int) -> float:
        """
        Compute Pinecone query cost.

        Args:
            queries: Number of queries

        Returns:
            Cost in dollars
        """
        return queries * self.PRICING['pinecone_query']

    def track_query_cost(
        self,
        query: str,
        contexts: list,
        response: str
    ) -> Dict[str, float]:
        """
        Track costs for a single query.

        Args:
            query: User query
            contexts: Retrieved contexts
            response: Generated response

        Returns:
            Dictionary with cost breakdown
        """
        # Count tokens
        query_tokens = self.count_tokens(query)
        context_text = "\n".join([c['text'] for c in contexts])
        context_tokens = self.count_tokens(context_text)
        response_tokens = self.count_tokens(response)

        # Compute costs
        embedding_cost = self.compute_embedding_cost(query_tokens)
        input_tokens = query_tokens + context_tokens
        llm_cost = self.compute_llm_cost(input_tokens, response_tokens)
        pinecone_cost = self.compute_pinecone_cost(1)

        total = embedding_cost + llm_cost + pinecone_cost

        # Update totals
        self.total_cost += total
        self.query_count += 1

        breakdown = {
            'embedding_cost': embedding_cost,
            'llm_cost': llm_cost,
            'pinecone_cost': pinecone_cost,
            'total_cost': total,
            'tokens': {
                'query': query_tokens,
                'context': context_tokens,
                'response': response_tokens,
                'total_input': input_tokens,
                'total_output': response_tokens
            }
        }

        logger.info(f"Query cost: ${total:.4f} (embedding: ${embedding_cost:.4f}, "
                   f"LLM: ${llm_cost:.4f}, Pinecone: ${pinecone_cost:.6f})")

        return breakdown

    def get_summary(self) -> Dict:
        """
        Get cost summary.

        Returns:
            Dictionary with cost statistics
        """
        avg_cost = self.total_cost / self.query_count if self.query_count > 0 else 0

        return {
            'total_cost': self.total_cost,
            'query_count': self.query_count,
            'average_cost_per_query': avg_cost,
            'projected_monthly_cost': avg_cost * 30 * 24 * 10,  # 10 queries/hour estimate
        }

    def reset(self):
        """Reset cost tracking."""
        self.total_cost = 0.0
        self.query_count = 0
        logger.info("Cost tracker reset")


# Global cost tracker instance
cost_tracker = CostTracker()
