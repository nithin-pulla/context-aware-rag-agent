"""Configuration management for the RAG system."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4-turbo-preview"

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "rag-tech-docs"

    # LangSmith Configuration (Optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "rag-tech-docs"

    # Application Configuration
    log_level: str = "INFO"
    max_retries: int = 3
    timeout_seconds: int = 30

    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval Configuration
    top_k: int = 5
    hybrid_alpha: float = 0.5
    rerank_top_n: int = 3

    # Evaluation Configuration
    batch_size: int = 10
    faithfulness_threshold: float = 0.7
    relevance_threshold: float = 0.75
    precision_threshold: float = 0.8
    recall_threshold: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
