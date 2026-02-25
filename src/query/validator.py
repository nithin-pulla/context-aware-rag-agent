"""Query validation and sanitization."""

import re
from typing import Tuple

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryValidator:
    """Validate and sanitize user queries."""

    MIN_LENGTH = 3
    MAX_LENGTH = 500

    # Patterns to detect potential prompt injection
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"system\s*:",
        r"<\|.*?\|>",  # Special tokens
    ]

    @classmethod
    def validate(cls, query: str) -> Tuple[bool, str]:
        """
        Validate query.

        Args:
            query: User query

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Strip whitespace
        query = query.strip()

        # Empty check
        if not query:
            return False, "Query cannot be empty"

        # Length checks
        if len(query) < cls.MIN_LENGTH:
            return False, f"Query too short (minimum {cls.MIN_LENGTH} characters)"

        if len(query) > cls.MAX_LENGTH:
            return False, f"Query too long (maximum {cls.MAX_LENGTH} characters)"

        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in query: {pattern}")
                return False, "Query contains potentially unsafe content"

        return True, ""

    @classmethod
    def sanitize(cls, query: str) -> str:
        """
        Sanitize query to prevent prompt injection.

        Args:
            query: User query

        Returns:
            Sanitized query
        """
        # Remove control characters except newline and tab
        query = ''.join(
            char for char in query
            if ord(char) >= 32 or char in ['\n', '\t']
        )

        # Limit consecutive newlines
        while '\n\n\n' in query:
            query = query.replace('\n\n\n', '\n\n')

        # Remove excessive whitespace
        query = ' '.join(query.split())

        return query.strip()

    @classmethod
    def process(cls, query: str) -> Tuple[bool, str, str]:
        """
        Validate and sanitize query.

        Args:
            query: User query

        Returns:
            Tuple of (is_valid, sanitized_query, error_message)
        """
        # First sanitize
        sanitized = cls.sanitize(query)

        # Then validate
        is_valid, error = cls.validate(sanitized)

        return is_valid, sanitized, error
