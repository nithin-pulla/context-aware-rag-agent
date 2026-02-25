"""Setup script for the RAG system package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="context-aware-rag",
    version="0.1.0",
    description="Production-grade RAG system for technical documentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.20",
        "langchain-openai>=0.1.7",
        "langchain-community>=0.0.38",
        "pinecone-client>=3.2.2",
        "pinecone-text>=0.7.1",
        "openai>=1.25.0",
        "tiktoken>=0.6.0",
        "ragas>=0.1.7",
        "datasets>=2.18.0",
        "pandas>=2.2.1",
        "numpy>=1.26.4",
        "python-dotenv>=1.0.1",
        "beautifulsoup4>=4.12.3",
        "markdown>=3.5.2",
        "pypdf>=4.1.0",
        "langsmith>=0.1.52",
        "prometheus-client>=0.20.0",
        "fastapi>=0.110.1",
        "uvicorn>=0.29.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.2.1",
        "tqdm>=4.66.2",
        "jsonlines>=4.0.0",
        "pyyaml>=6.0.1",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-asyncio>=0.23.6",
            "pytest-cov>=5.0.0",
            "black>=24.3.0",
            "ruff>=0.3.5",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-ingest=scripts.ingest:main",
            "rag-query=scripts.query:main",
            "rag-evaluate=scripts.evaluate:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
