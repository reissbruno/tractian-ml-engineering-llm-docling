"""
Markdown content cleaning service before chunking.
Removes unwanted elements like indices, contacts, page numbers, etc.
"""

import re

from src.logger import logger


def clean_markdown_content(markdown_text: str) -> str:
    """
    Cleans markdown content removing unwanted elements.

    Removes:
    - Initial index (## Index and TOC table)
    - Final contacts (phone, email, "Subject to change")
    - Lines with only page number or ---
    - Blocks of "Images on page X..."
    - Excessive HTML comments

    Args:
        markdown_text: Raw markdown text

    Returns:
        Cleaned markdown text
    """
    lines = markdown_text.split('\n')
    cleaned_lines = []

    skip_until_next_section = False
    in_index_section = False
    in_contact_section = False

    in_table_index = False  # To detect indices in table format (Docling)

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect start of index (multiple formats)
        # Also detects "Index" without markdown
        if re.match(r'^(#{1,6}\s+)?(Índice|ÍNDICE|Sumário|SUMÁRIO|Table of Contents|Contents|Index|TOC)\s*$', stripped, re.IGNORECASE):
            in_index_section = True
            skip_until_next_section = True
            in_table_index = True  # Next lines may be table
            continue

        # Detect Docling index tables (start with | and have dotted numbers)
        # Example: | 1.1 | Electric Motors ....................................................6 |
        # Example: |----------------------------------------------------------------------------------------------------------|
        if stripped.startswith('|') and in_table_index:
            # Check if it's a table row with index content
            if re.search(r'[\d\.]+\s+.+?\.{3,}\s*\d+', stripped) or stripped.count('-') > 10:
                continue

        # Detect table cells that are part of index
        # Detect lines with only dotted numbers (1.1, 1.2.3, etc)
        if in_table_index and re.match(r'^\|\s*[\d\.]+\s*\|', stripped):
            continue

        # Detect typical index lines (page number followed by dots)
        # Example: "1.1 Introduction ............................. 5"
        # Example: "Chapter 1 .............................. 10"
        if re.match(r'^[\d\.]+\s+.+?\.{3,}\s*\d+$', stripped):
            if not in_index_section:
                in_index_section = True
                skip_until_next_section = True
            continue

        # Detect index lines without dots
        # Example: "1.1 Introduction                         5"
        if re.match(r'^[\d\.]+\s+[\w\s]+\s{3,}\d+$', stripped):
            if not in_index_section:
                in_index_section = True
                skip_until_next_section = True
            continue

        # Detect index pattern with tabs (very common in PDFs)
        # Example: "1.1		Electric Motors.....................................................6"
        # Example: "1.1 Introduction\t\t\t5"
        # Example: "### 5.1\n\tPole Number Variation ...............................30"
        if re.match(r'^(#{1,6}\s+)?[\d\.]+[\s\t]+.+?[\.\s\t]+\d+$', stripped):
            if not in_index_section:
                in_index_section = True
                skip_until_next_section = True
            continue

        # Detect markdown header followed by index line (WEG pattern)
        # Example: "### 5.1" followed by "\tPole Number Variation ....30"
        # Example: "### 11.1	 		 	 Motor Type Selection for Different"
        # If it has markdown + number + multiple tabs, it's an index
        if re.match(r'^#{1,6}\s+[\d\.]+', stripped):
            # Se tem tabs múltiplos ou se é só número, é índice
            if '\t\t' in line or re.match(r'^#{1,6}\s+[\d\.]+\s*$', stripped):
                if not in_index_section:
                    in_index_section = True
                    skip_until_next_section = True
                continue

        # Detect index entries starting with letter/word followed by page
        # Example: "Introduction ...................... 3"
        # Example: "A. Glossary .................... 50"
        # Example: "Figure 1 - Diagram ...... 5"
        if re.match(r'^([A-Z][\.\)]|Figura|Tabela|Table|Figure)\s+.+?\.{3,}\s*\d+$', stripped, re.IGNORECASE):
            if not in_index_section:
                in_index_section = True
                skip_until_next_section = True
            continue

        # Detect common section header before index
        # Example: "CONTENTS", "CONTENTS", "LIST OF FIGURES"
        # Example: "## LIST OF FIGURES"
        if re.match(r'^#{0,6}\s*(CONTEÚDO|CONTENTS?|LISTA DE (FIGURAS?|TABELAS?)|LIST OF (FIGURES?|TABLES?))\s*$', stripped, re.IGNORECASE):
            in_index_section = True
            skip_until_next_section = True
            continue

        # Detect index continuation lines (multi-line)
        # Example: "		 	 Loads...............................................................57"
        # Lines that start with tabs/spaces and end with dots + number
        if re.match(r'^[\s\t]+.+?\.{3,}\s*\d+$', line):
            if not in_index_section:
                in_index_section = True
                skip_until_next_section = True
            continue

        # Detect contacts/final information section
        if re.search(r'(sujeito a alterações|subject to change|atendimento|contato|tel\.|telefone|e-mail|www\.)', stripped, re.IGNORECASE):
            in_contact_section = True
            continue

        # Exit skip mode when finding new real section (not index)
        # New section starts with ## followed by text that is NOT just a number
        if skip_until_next_section:
            # Detect real section (not index)
            if re.match(r'^#{1,6}\s+(?!\d+\.?\s*$)(?!Índice|Sumário|Table|Contents|Index|TOC)', stripped, re.IGNORECASE):
                # Check if it's not a disguised index line
                # Indices have final dots (...57) or multiple tabs
                if not re.search(r'\.{3,}\s*\d+$', stripped) and '\t\t' not in stripped:
                    skip_until_next_section = False
                    in_index_section = False
                    in_table_index = False  # Exit index table mode too

        # Skip lines during index
        if in_index_section or skip_until_next_section:
            continue

        # Skip contact lines
        if in_contact_section:
            continue

        # Remove HTML comments about images
        if re.match(r'<!--\s*Imagens? na página', stripped, re.IGNORECASE):
            continue

        # Remove lines that are just "Images on page X:"
        if re.match(r'\*\*Imagens na página \d+:\*\*', stripped):
            continue

        # Remove individual image list lines
        if re.match(r'-\s+(raster|vector):\s*`[^`]+`', stripped):
            continue

        # Remove lines with only page number (isolated)
        if re.match(r'^\d{1,4}$', stripped):
            continue

        # Remove lines that are just separators ---
        if re.match(r'^-{3,}$', stripped):
            continue

        # Remove lines that are just ===
        if re.match(r'^={3,}$', stripped):
            continue

        # Remove metadata HTML comments (except page separators)
        if stripped.startswith('<!--') and 'Página' not in stripped and 'Page' not in stripped:
            if 'Gerado de' in stripped or 'Generated from' in stripped or 'Páginas:' in stripped or 'Pages:' in stripped or 'fonte mediana' in stripped or 'median font' in stripped:
                continue

        # Remove consecutive empty lines (keep only one)
        if not stripped:
            if cleaned_lines and not cleaned_lines[-1].strip():
                continue

        cleaned_lines.append(line)

    # Join lines and do final cleanups
    cleaned_text = '\n'.join(cleaned_lines)

    # Remove multiple consecutive empty lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # Remove whitespace at end of lines
    cleaned_text = '\n'.join(line.rstrip() for line in cleaned_text.split('\n'))

    # Safety check: if we removed more than 90% of content, return original
    # This prevents over-aggressive cleaning
    if len(cleaned_text) < len(markdown_text) * 0.1:
        logger.warning(f"Content cleaning removed >90% of content ({len(markdown_text)} -> {len(cleaned_text)}). Returning original.")
        return markdown_text.strip()

    logger.debug(f"Content cleaning: {len(markdown_text)} -> {len(cleaned_text)} characters")

    return cleaned_text.strip()


def extract_metadata_from_markdown(markdown_text: str) -> dict:
    """
    Extracts metadata from markdown (pages, document name, etc).

    Args:
        markdown_text: Markdown text

    Returns:
        Dict with extracted metadata
    """
    metadata = {
        'total_pages': 0,
        'source_file': None,
        'median_font_size': None
    }

    # Search for metadata comments
    for line in markdown_text.split('\n')[:10]:  # First 10 lines
        # <!-- Generated from file.pdf -->
        match = re.search(r'<!--\s*(Gerado de|Generated from)\s+(.+?)\s+usando', line)
        if match:
            metadata['source_file'] = match.group(2).strip()

        # <!-- Pages: 120, median font: 10.5 -->
        match = re.search(r'<!--\s*(Páginas|Pages):\s*(\d+)', line)
        if match:
            metadata['total_pages'] = int(match.group(2))

        match = re.search(r'(fonte mediana|median font):\s*([\d.]+)', line)
        if match:
            metadata['median_font_size'] = float(match.group(2))

    return metadata


def split_by_pages(markdown_text: str) -> list[tuple[int, str]]:
    """
    Splits markdown by pages.

    Args:
        markdown_text: Complete markdown text

    Returns:
        List of tuples (page_number, page_content)
    """
    pages = []
    current_page = None
    current_content = []

    for line in markdown_text.split('\n'):
        # Detect page marker: <!-- Page 13 -->
        match = re.match(r'<!--\s*(Página|Page)\s+(\d+)\s*-->', line.strip())
        if match:
            # Save previous page if exists
            if current_page is not None and current_content:
                pages.append((current_page, '\n'.join(current_content).strip()))

            # Start new page
            current_page = int(match.group(2))
            current_content = []
            continue

        # Skip page separators ---
        if line.strip() == '---':
            continue

        # Add line to current page content
        if current_page is not None:
            current_content.append(line)

    # Save last page
    if current_page is not None and current_content:
        pages.append((current_page, '\n'.join(current_content).strip()))

    # If no pages were detected (Docling doesn't generate markers),
    # treat all content as page 1
    if not pages and markdown_text.strip():
        logger.info("No page markers detected. Treating all content as single page.")
        pages.append((1, markdown_text.strip()))

    return pages
