"""
Document structure analyzer for automatic type detection.
Classifies documents by structural characteristics (not by keywords).
"""

import re
from statistics import mean, median
from typing import Dict

from src.logger import logger


class DocumentStructureAnalyzer:
    """
    Analyzes document structure to determine its type
    and the best chunking configurations.
    """

    def __init__(self):
        # Thresholds for small manual/instruction classification
        self.SMALL_DOC_CHARS = 10000      # Reduced to better detect small manuals
        self.AVG_PARA_SHORT = 600          # Really short paragraph average
        self.MAX_PARA_MODERATE = 1500      # Reduced
        self.SHORT_PARA_RATIO_MIN = 0.65   # Increased: majority should be short
        self.SHORT_PARA_THRESHOLD = 350    # Lower threshold for "short" paragraph

    def analyze_document_structure(self, text: str, num_pages: int = None) -> Dict:
        """
        Analyzes document structure and extracts metrics.

        Args:
            text: Complete document text (markdown)
            num_pages: Number of pages (optional)

        Returns:
            Dict with structural metrics
        """
        total_chars = len(text)

        # Split into paragraphs (blocks separated by double empty line)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        num_paragraphs = len(paragraphs)

        # Paragraph lengths
        para_lens = [len(p) for p in paragraphs] or [0]
        avg_para_len = mean(para_lens) if para_lens else 0
        median_para_len = median(para_lens) if para_lens else 0
        max_para_len = max(para_lens) if para_lens else 0
        min_para_len = min(para_lens) if para_lens else 0

        # Proportion of short paragraphs (instructions/commands)
        short_paras = sum(1 for l in para_lens if l <= self.SHORT_PARA_THRESHOLD)
        short_para_ratio = short_paras / max(1, num_paragraphs)

        # Detect headings (##, ###, etc.)
        headings = re.findall(r"^#{1,6}\s+.+$", text, re.MULTILINE)
        num_headings = len(headings)

        # Headings/paragraphs ratio
        heading_density = num_headings / max(1, num_paragraphs)

        # Detect lists (-, *, numbers)
        list_items = re.findall(r"^[\s]*[\-\*•]\s+.+$|^[\s]*\d+[\.\)]\s+.+$", text, re.MULTILINE)
        num_list_items = len(list_items)
        list_density = num_list_items / max(1, num_paragraphs)

        # Detect tables (markdown)
        tables = re.findall(r"\|.*\|", text)
        num_tables = len(tables)

        stats = {
            "total_chars": total_chars,
            "num_pages": num_pages or 1,
            "num_paragraphs": num_paragraphs,
            "avg_para_len": avg_para_len,
            "median_para_len": median_para_len,
            "max_para_len": max_para_len,
            "min_para_len": min_para_len,
            "short_para_ratio": short_para_ratio,
            "num_headings": num_headings,
            "heading_density": heading_density,
            "num_list_items": num_list_items,
            "list_density": list_density,
            "num_tables": num_tables,
        }

        logger.info(
            f"Structural analysis: {total_chars} chars, {num_pages or '?'} pages, "
            f"{num_paragraphs} paragraphs (avg={avg_para_len:.0f}), "
            f"short_ratio={short_para_ratio:.2f}, headings={num_headings}"
        )

        return stats

    def is_small_instruction_manual(self, stats: Dict) -> bool:
        """
        Detects if it's a small instruction manual (like LB5001).

        Characteristics:
        - Small document (< 10k chars AND < 5 pages)
        - Short paragraphs on average (< 600 chars)
        - No giant sections (max < 1500 chars)
        - Majority of paragraphs are short (>= 65% with < 350 chars)

        Args:
            stats: Dictionary of structural statistics

        Returns:
            True if it's a small manual/instruction
        """
        # Criterion 1: Really small document (all conditions)
        is_very_small = (
            stats["total_chars"] <= self.SMALL_DOC_CHARS  # <= 10k chars
            and stats["num_pages"] <= 5
            and stats["avg_para_len"] <= self.AVG_PARA_SHORT  # <= 600
            and stats["max_para_len"] <= self.MAX_PARA_MODERATE  # <= 1500
            and stats["short_para_ratio"] >= self.SHORT_PARA_RATIO_MIN  # >= 65%
        )

        # Criterion 2: Small document with instructions (more flexible)
        # Detects documents with high proportion of short paragraphs
        is_instruction_style = (
            stats["total_chars"] <= 10000  # Up to 10k chars
            and stats["num_pages"] <= 5
            and stats["short_para_ratio"] >= 0.75  # 75%+ short paragraphs = instructions
        )

        is_small = is_very_small or is_instruction_style

        if is_small:
            logger.info(
                "Document classified as: SMALL MANUAL/INSTRUCTIONS "
                f"(pages={stats['num_pages']}, chars={stats['total_chars']}, "
                f"avg_para={stats['avg_para_len']:.0f}, short_ratio={stats['short_para_ratio']:.2f})"
            )
        else:
            logger.info(
                "Document classified as: LARGE CATALOG/TECHNICAL GUIDE "
                f"(pages={stats['num_pages']}, chars={stats['total_chars']}, "
                f"avg_para={stats['avg_para_len']:.0f})"
            )

        return is_small

    def is_highly_structured(self, stats: Dict) -> bool:
        """
        Detects if it's a highly structured document (many lists, tables, headings).

        Args:
            stats: Dictionary of structural statistics

        Returns:
            True if highly structured
        """
        return (
            stats["heading_density"] >= 0.3  # 30%+ of paragraphs are headings
            or stats["list_density"] >= 0.5  # 50%+ have lists
            or stats["num_tables"] >= 10      # Many tables
        )

    def get_chunking_config(self, text: str, num_pages: int = None) -> Dict:
        """
        Automatically determines the best chunking configuration
        based on document structure.

        Args:
            text: Complete document text
            num_pages: Number of pages (optional)

        Returns:
            Dict with chunking configuration:
            - chunk_size: Chunk size
            - chunk_overlap: Overlap between chunks
            - strategy: Strategy ('small_manual', 'structured', 'default')
        """
        # Analyze structure
        stats = self.analyze_document_structure(text, num_pages)

        # Decision based on structure
        if self.is_small_instruction_manual(stats):
            # Small manual: smaller chunks to maintain instruction context
            config = {
                "chunk_size": 500,      # 300-700 chars
                "chunk_overlap": 100,   # 20% overlap
                "strategy": "small_manual",
                "description": "Small instruction manual"
            }

        elif self.is_highly_structured(stats):
            # Structured document: moderate chunks
            config = {
                "chunk_size": 1200,
                "chunk_overlap": 200,
                "strategy": "structured",
                "description": "Highly structured document (lists/tables)"
            }

        else:
            # Large technical catalog/guide: large chunks
            config = {
                "chunk_size": 2200,
                "chunk_overlap": 350,
                "strategy": "default",
                "description": "Catalog or extensive technical guide"
            }

        logger.info(
            f"Chunking configuration selected: {config['strategy']} "
            f"(chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}) "
            f"- {config['description']}"
        )

        return config


# Singleton to avoid recreating instance
_analyzer_instance = None


def get_document_analyzer() -> DocumentStructureAnalyzer:
    """
    Returns singleton instance of document analyzer.

    Returns:
        DocumentStructureAnalyzer
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DocumentStructureAnalyzer()
    return _analyzer_instance
