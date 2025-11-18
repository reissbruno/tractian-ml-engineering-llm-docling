import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.logger import logger


# =======================================
#   Providers / LLMService
# =======================================

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMService:
    """
    Service that encapsulates the LLM (OpenAI GPT or Google Gemini),
    providing methods for generating responses with or without images (multimodal).
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        self.provider = provider
        self.temperature = temperature

        if provider == LLMProvider.OPENAI:
            self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                **kwargs
            )
        elif provider == LLMProvider.GEMINI:
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temperature,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info(
            f"LLMService initialized: provider={provider.value}, "
            f"model={self.model_name}, temperature={temperature}"
        )

    def _build_messages(
        self,
        question: str,
        context_chunks: List[str],
        images: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Build messages for the LLM.

        Args:
            question: User question
            context_chunks: List of context texts
            images: Optional list of base64 images

        Returns:
            List of messages (SystemMessage, HumanMessage)
        """
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "You are an assistant specialized in technical documentation. "
            "Answer questions based only on the provided context. "
            "If the answer is not in the context, say so clearly. "
            "Be precise, objective, and cite specific parts of the context when possible."
        )

        messages = [SystemMessage(content=system_prompt)]

        if images:
            content_parts = [
                {
                    "type": "text",
                    "text": (
                        f"Context:\n{context_text}\n\n"
                        f"Question: {question}\n\n"
                        "Answer based on the context and images provided:"
                    )
                }
            ]

            for img_base64 in images[:5]:
                if img_base64.startswith("data:image"):
                    image_url = img_base64
                else:
                    image_url = f"data:image/jpeg;base64,{img_base64}"

                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

            messages.append(HumanMessage(content=content_parts))
        else:
            user_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the provided context:"
            )
            messages.append(HumanMessage(content=user_prompt))

        return messages

    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        images: Optional[List[str]] = None
    ) -> str:
        """
        Generate an answer using the LLM (synchronous).

        Args:
            question: User question
            context_chunks: List of context texts
            images: Optional list of base64 images

        Returns:
            Generated answer
        """
        messages = self._build_messages(question, context_chunks, images)

        logger.info(
            f"Generating answer with {len(context_chunks)} chunks, "
            f"{len(images) if images else 0} images"
        )

        try:
            response = self.llm.invoke(messages)
            answer = response.content

            logger.info(f"Answer generated: {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            raise

    async def generate_answer_stream(
        self,
        question: str,
        context_chunks: List[str],
        images: Optional[List[str]] = None
    ):
        """
        Generate an answer using the LLM (asynchronous streaming).

        Args:
            question: User question
            context_chunks: List of context texts
            images: Optional list of base64 images

        Yields:
            Answer chunks as they are generated
        """
        messages = self._build_messages(question, context_chunks, images)

        logger.info(
            f"Generating streaming answer with {len(context_chunks)} chunks, "
            f"{len(images) if images else 0} images"
        )

        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            raise


# Singleton pattern
_llm_service = None


def get_llm_service(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.4,
) -> LLMService:
    """
    Get or create LLMService instance (singleton pattern).

    Args:
        provider: LLM provider ("openai" or "gemini")
        model_name: Model name (optional)
        temperature: Temperature for generation

    Returns:
        LLMService instance
    """
    global _llm_service

    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        logger.warning(f"Invalid provider '{provider}', defaulting to OpenAI")
        provider_enum = LLMProvider.OPENAI

    if _llm_service is None or _llm_service.provider != provider_enum:
        _llm_service = LLMService(
            provider=provider_enum,
            model_name=model_name,
            temperature=temperature,
        )

    return _llm_service
