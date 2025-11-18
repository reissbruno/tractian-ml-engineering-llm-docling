"""
Intelligent chunking service with overlap and adaptive strategy.
"""

import re
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import logger


class SmartChunker:
    """
    Intelligent chunker that uses different strategies based on content.
    """

    def __init__(
        self,
        chunk_size: int = 2200,
        chunk_overlap: int = 350,
        separators: List[str] = None
    ):
        """
        Initializes the chunker.

        Args:
            chunk_size: Target chunk size in characters (default: 2200)
            chunk_overlap: Overlap between chunks in characters (default: 350, ~15%)
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Smart separators (priority order)
        if separators is None:
            separators = [
                "\n## ",      # Large section
                "\n### ",     # Subsection
                "\n#### ",    # Sub-subsection
                "\nTabela ",  # Table start
                "\nFigura ",  # Figure caption
                "\n\n",       # Paragraph
                "\n",         # Line
                ". ",         # Sentence
                " ",          # Word
                ""            # Character
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )

        logger.info(
            f"SmartChunker initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap} (~{chunk_overlap/chunk_size*100:.1f}%)"
        )

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Splits text into chunks with metadata.

        Args:
            text: Text to be split
            metadata: Base metadata to include in each chunk

        Returns:
            List of dicts with 'content' and 'metadata'
        """
        if not text or not text.strip():
            return []

        chunks = self.splitter.split_text(text)

        result = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_index': idx,
                'chunk_size': len(chunk_text),
            })

            result.append({
                'content': chunk_text.strip(),
                'metadata': chunk_metadata
            })

        logger.debug(f"Text of {len(text)} chars split into {len(result)} chunks")
        return result

    def chunk_by_pages(
        self,
        pages: List[tuple],
        base_metadata: Dict = None
    ) -> List[Dict]:
        """
        Splits content by pages, creating chunks within each page.

        Args:
            pages: List of tuples (page_number, page_content)
            base_metadata: Base document metadata

        Returns:
            List of chunks with metadata including page number
        """
        all_chunks = []

        for page_num, page_content in pages:
            if not page_content or not page_content.strip():
                continue

            page_metadata = base_metadata.copy() if base_metadata else {}
            page_metadata['page_number'] = page_num

            page_chunks = self.chunk_text(page_content, page_metadata)

            for chunk in page_chunks:
                chunk['metadata']['page_start'] = page_num
                chunk['metadata']['page_end'] = page_num

            all_chunks.extend(page_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks

    def detect_content_type(self, text: str) -> str:
        """
        Detects the content type of the chunk.

        Args:
            text: Chunk text

        Returns:
            Type: 'text', 'list', 'formula', 'table_caption'
        """
        # Tables: multiple lines with pipes |
        if text.count('|') >= 6 and text.count('\n') >= 3:
            return 'table'

        # Lists: multiple lines starting with - or numbers
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 3:
            list_lines = sum(
                1 for l in lines
                if re.match(r'^[\-\*•]\s+', l) or re.match(r'^\d+[\.\)]\s+', l)
            )
            if list_lines >= len(lines) * 0.6:
                return 'list'

        # Formulas: contains mathematical symbols
        if re.search(r'[=≈≤≥∑∫√]', text):
            return 'formula'

        # Figure caption
        if re.search(r'^\s*(Figura|Tabela|Fig\.|Tab\.)\s+\d+', text, re.IGNORECASE):
            return 'caption'

        return 'text'

    def enrich_chunks_with_content_type(self, chunks: List[Dict]) -> List[Dict]:
        """
        Adds content type detection to chunks.

        Args:
            chunks: List of chunks

        Returns:
            Chunks enriched with content_type
        """
        for chunk in chunks:
            content_type = self.detect_content_type(chunk['content'])
            chunk['metadata']['content_type'] = content_type

        return chunks


def create_chunker(
    chunk_size: int = 2200,
    chunk_overlap: int = 350
) -> SmartChunker:
    """
    Factory function to create a chunker.

    Args:
        chunk_size: Chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        SmartChunker instance
    """
    return SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
