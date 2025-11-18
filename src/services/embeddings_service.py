"""
Service for generating embeddings using OpenAI text-embedding-3-large.
"""

import asyncio
import os
from typing import List

from langchain_openai import OpenAIEmbeddings

from src.logger import logger


class EmbeddingsService:
    """
    Embeddings service using OpenAI text-embedding-3-large.
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initializes the embeddings service.

        Args:
            model_name: OpenAI model name (default: text-embedding-3-large)
        """
        self.model_name = model_name
        self.embeddings = None
        logger.info(f"Initializing EmbeddingsService with OpenAI model: {model_name}")

    def load_model(self):
        """
        Initializes the OpenAI embeddings client (lazy loading).
        """
        if self.embeddings is None:
            logger.info(f"Initializing OpenAI Embeddings: {self.model_name}")
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is not defined in environment variables")

                self.embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    api_key=api_key
                )
                logger.info("OpenAI Embeddings initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI Embeddings: {e}")
                raise

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.load_model()

        try:
            embeddings_list = self.embeddings.embed_documents(texts)

            logger.info(f"Generated {len(embeddings_list)} OpenAI embeddings of dimension {len(embeddings_list[0])}")
            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise

    def encode_single(self, text: str) -> List[float]:
        """
        Generates embedding for a single text (query).

        Args:
            text: Text to generate embedding

        Returns:
            Embedding vector
        """
        self.load_model()

        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Query embedding generated: {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    async def encode_async(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Asynchronous version of encode() with batching to avoid rate limits.
        Processes texts in batches to respect OpenAI's 40k tokens/min limit.

        Args:
            texts: List of texts to generate embeddings
            batch_size: Number of texts per batch (default: 100)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.load_model()

        if len(texts) <= batch_size:
            try:
                embeddings_list = await self.embeddings.aembed_documents(texts)
                logger.info(f"Generated {len(embeddings_list)} OpenAI embeddings (async) of dimension {len(embeddings_list[0])}")
                return embeddings_list
            except Exception as e:
                logger.error(f"Error generating asynchronous embeddings with OpenAI: {e}")
                raise

        logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)

                logger.info(f"Batch {batch_num}/{total_batches} completed: {len(batch_embeddings)} embeddings")

                # Delay between batches to avoid rate limit (only if not the last batch)
                if i + batch_size < len(texts):
                    await asyncio.sleep(1.5)  # 1.5 seconds between batches

            except Exception as e:
                logger.error(f"Error in batch {batch_num}/{total_batches}: {e}")
                raise

        logger.info(f"Total embeddings generated: {len(all_embeddings)} of dimension {len(all_embeddings[0])}")
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """
        Returns the embedding dimension of the model.

        Returns:
            Dimension of embedding vectors
        """
        # text-embedding-3-large tem dimensão 3072
        # text-embedding-3-small tem dimensão 1536
        if "large" in self.model_name:
            return 3072
        elif "small" in self.model_name:
            return 1536
        else:
            # Fallback: generate a test embedding to discover
            self.load_model()
            test_embedding = self.encode_single("test")
            return len(test_embedding)


# Global instance (singleton) of the embeddings service
_embeddings_service = None


def get_embeddings_service() -> EmbeddingsService:
    """
    Returns the singleton instance of the embeddings service.

    Returns:
        EmbeddingsService instance
    """
    global _embeddings_service

    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
        # Load model on initialization
        _embeddings_service.load_model()

    return _embeddings_service
