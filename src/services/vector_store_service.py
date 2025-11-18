"""
Vector store service using ChromaDB with structured metadata.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.logger import logger


class VectorStoreService:
    """
    Vector store service using ChromaDB.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initializes the vector store service.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}

        logger.info(f"Initializing VectorStoreService: persist_directory={persist_directory}")

        os.makedirs(persist_directory, exist_ok=True)

    def get_client(self) -> chromadb.Client:
        """
        Returns the ChromaDB client (lazy loading).

        Returns:
            ChromaDB client
        """
        if self.client is None:
            logger.info("Connecting to ChromaDB...")
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Connected to ChromaDB")

        return self.client

    def get_or_create_collection(self, user_id: int):
        """
        Gets or creates a collection for a user.

        Args:
            user_id: User ID

        Returns:
            ChromaDB collection
        """
        collection_name = f"user_{user_id}_documents"

        if collection_name not in self.collections:
            client = self.get_client()

            try:
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"user_id": str(user_id)}
                )
                self.collections[collection_name] = collection
                logger.info(f"Collection '{collection_name}' obtained/created")

            except Exception as e:
                logger.error(f"Error getting/creating collection: {e}")
                raise

        return self.collections[collection_name]

    def add_chunks(
        self,
        user_id: int,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Adds chunks to the vector store.

        Args:
            user_id: User ID
            chunks: List of chunks with 'content' and 'metadata'
            embeddings: List of corresponding embeddings

        Returns:
            Number of chunks added
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) != embeddings ({len(embeddings)})")

        collection = self.get_or_create_collection(user_id)

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = chunk['metadata'].get('doc_id', 'unknown')
            page = chunk['metadata'].get('page_number', 0)
            chunk_idx = chunk['metadata'].get('chunk_index', idx)

            chunk_id = f"{doc_id}_p{page}_c{chunk_idx}"

            # Prepare metadata (ChromaDB accepts only basic types)
            metadata = {
                'doc_id': str(chunk['metadata'].get('doc_id', '')),
                'doc_name': str(chunk['metadata'].get('doc_name', '')),
                'page_start': int(chunk['metadata'].get('page_start', 0)),
                'page_end': int(chunk['metadata'].get('page_end', 0)),
                'page_number': int(chunk['metadata'].get('page_number', 0)),
                'chunk_index': int(chunk['metadata'].get('chunk_index', 0)),
                'chunk_size': int(chunk['metadata'].get('chunk_size', 0)),
                'content_type': str(chunk['metadata'].get('content_type', 'text')),
                'language': str(chunk['metadata'].get('language', 'pt-BR')),
                'parser': str(chunk['metadata'].get('parser', 'unknown')),
                'has_images': str(chunk['metadata'].get('has_images', 'false')),
                'image_ids': str(chunk['metadata'].get('image_ids', '')),
                'ingestion_ts': datetime.utcnow().isoformat(),
                'version': str(chunk['metadata'].get('version', '1'))
            }

            ids.append(chunk_id)
            documents.append(chunk['content'])
            metadatas.append(metadata)
            embeddings_list.append(embedding)

        try:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )

            logger.info(f"Added {len(ids)} chunks to collection '{collection.name}'")
            return len(ids)

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self,
        user_id: int,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Searches for similar chunks.

        Args:
            user_id: User ID
            query_embedding: Query embedding
            n_results: Number of results to return
            filter_metadata: Metadata filters (optional)

        Returns:
            Search results
        """
        collection = self.get_or_create_collection(user_id)

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata if filter_metadata else None
            )

            logger.debug(f"Search returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Error in vector store search: {e}")
            raise

    def delete_document(self, user_id: int, doc_id: str) -> int:
        """
        Deletes all chunks of a document.

        Args:
            user_id: User ID
            doc_id: Document ID

        Returns:
            Number of chunks deleted
        """
        collection = self.get_or_create_collection(user_id)

        try:
            results = collection.get(
                where={"doc_id": doc_id}
            )

            if results['ids']:
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks from document {doc_id}")
                return len(results['ids'])
            else:
                logger.info(f"No chunks found for document {doc_id}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting document from vector store: {e}")
            raise

    def get_collection_stats(self, user_id: int) -> Dict:
        """
        Returns statistics of the user's collection.

        Args:
            user_id: User ID

        Returns:
            Dict with statistics
        """
        collection = self.get_or_create_collection(user_id)

        try:
            count = collection.count()
            return {
                'collection_name': collection.name,
                'total_chunks': count,
                'user_id': user_id
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'collection_name': f"user_{user_id}_documents",
                'total_chunks': 0,
                'user_id': user_id,
                'error': str(e)
            }


# Global (singleton) instance of vector store service
_vector_store_service = None


def get_vector_store_service() -> VectorStoreService:
    """
    Returns the singleton instance of the vector store service.

    Returns:
        VectorStoreService instance
    """
    global _vector_store_service

    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()

    return _vector_store_service
