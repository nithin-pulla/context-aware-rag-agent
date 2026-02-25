"""Pinecone client for vector indexing and search."""

import os
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec

from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class PineconeClient:
    """Manage Pinecone index operations."""

    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize Pinecone client.

        Args:
            index_name: Name of the Pinecone index
        """
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = index_name or settings.pinecone_index_name
        self.index = None

    def create_index(self, dimension: int = 1536, metric: str = "dotproduct"):
        """
        Create a new Pinecone index.

        Args:
            dimension: Vector dimension (1536 for text-embedding-3-small)
            metric: Distance metric (dotproduct, cosine, euclidean)
        """
        try:
            # Check if index exists
            if self.index_name in self.pc.list_indexes().names():
                logger.info(f"Index {self.index_name} already exists")
                return

            logger.info(f"Creating index {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment
                )
            )
            logger.info(f"Successfully created index {self.index_name}")

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def connect(self):
        """Connect to existing index."""
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index {self.index_name}")
            return self.index

        except Exception as e:
            logger.error(f"Error connecting to index: {e}")
            raise

    def upsert_chunks(
        self,
        chunks: List[Dict],
        namespace: str = "default",
        batch_size: int = 100
    ):
        """
        Upsert chunks to Pinecone index.

        Args:
            chunks: List of chunks with embeddings
            namespace: Namespace for isolation
            batch_size: Batch size for upsert operations
        """
        if not self.index:
            self.connect()

        logger.info(f"Upserting {len(chunks)} chunks to namespace '{namespace}'")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vectors = []

            for chunk in batch:
                vector = {
                    'id': chunk['chunk_id'],
                    'values': chunk['dense_embedding'],
                    'metadata': {
                        'text': chunk['text'][:1000],  # Pinecone metadata limit
                        'source': chunk['source'],
                        'chunk_index': chunk['chunk_index'],
                    }
                }

                # Add sparse values if available
                if 'sparse_embedding' in chunk:
                    vector['sparse_values'] = {
                        'indices': chunk['sparse_embedding']['indices'],
                        'values': chunk['sparse_embedding']['values']
                    }

                vectors.append(vector)

            # Upsert batch
            try:
                self.index.upsert(vectors=vectors, namespace=namespace)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                raise

        logger.info(f"Successfully upserted {len(chunks)} chunks")

    def query(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict] = None,
        top_k: int = 10,
        namespace: str = "default",
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query Pinecone index with hybrid search.

        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Sparse embedding vector (optional)
            top_k: Number of results to return
            namespace: Namespace to query
            filter: Metadata filter
            include_metadata: Include metadata in results

        Returns:
            List of matches
        """
        if not self.index:
            self.connect()

        try:
            query_params = {
                'vector': dense_vector,
                'top_k': top_k,
                'namespace': namespace,
                'include_metadata': include_metadata
            }

            # Add sparse vector if provided
            if sparse_vector:
                query_params['sparse_vector'] = sparse_vector

            # Add filter if provided
            if filter:
                query_params['filter'] = filter

            results = self.index.query(**query_params)
            return results.get('matches', [])

        except Exception as e:
            logger.error(f"Error querying index: {e}")
            raise

    def delete_namespace(self, namespace: str):
        """
        Delete all vectors in a namespace.

        Args:
            namespace: Namespace to delete
        """
        if not self.index:
            self.connect()

        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace '{namespace}'")

        except Exception as e:
            logger.error(f"Error deleting namespace: {e}")
            raise

    def get_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats
        """
        if not self.index:
            self.connect()

        try:
            stats = self.index.describe_index_stats()
            return {
                'dimension': stats.get('dimension'),
                'total_vector_count': stats.get('total_vector_count'),
                'namespaces': stats.get('namespaces', {})
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
