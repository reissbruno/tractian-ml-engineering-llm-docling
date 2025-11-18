"""
Fast PDF processor using PyMuPDF (fitz).

Alternative to Docling for fast PDF → Markdown conversion.
- Much faster (no heavy ML models)
- Simpler but still structured output
- Ideal for text-predominant documents
"""

import asyncio
import base64
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import fitz  # PyMuPDF

from src.logger import logger

KEEP_DETAILED_IMAGE_LIST = False

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=2)

# =====================================================================
#  Font analysis helpers
# =====================================================================

def collect_font_sizes(doc):
    """
    Collects all font sizes used in the document.
    Used to infer headings (larger font) vs normal text.
    """
    sizes = []
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size")
                    if size:
                        sizes.append(size)
    return sizes


def build_heading_levels(font_sizes, max_levels=3):
    """
    Creates a mapping font_size → heading level (1,2,3...).
    The largest font sizes become higher-level headings.
    """
    if not font_sizes:
        return {}

    unique_sizes = sorted(set(font_sizes), reverse=True)
    top_sizes = unique_sizes[:max_levels]

    levels = {}
    for i, size in enumerate(top_sizes, start=1):
        levels[size] = i  # 1 = H1, 2 = H2, 3 = H3...
    return levels


def block_average_font_size(block):
    """Calculates the average font size of a block."""
    sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            size = span.get("size")
            if size:
                sizes.append(size)
    return sum(sizes) / len(sizes) if sizes else 0.0


# =====================================================================
#  Text / block helpers
# =====================================================================

def block_text(block):
    """
    Extracts text from a PyMuPDF block with simple inline formatting:
    - **bold**
    - *italic*
    """
    parts = []
    for line in block.get("lines", []):
        spans_text = []
        for span in line.get("spans", []):
            text = span.get("text", "")
            font = (span.get("font") or "").lower()
            flags = span.get("flags", 0)

            # Simple style heuristics
            is_bold = ("bold" in font) or (flags & 2)
            is_italic = ("italic" in font) or ("oblique" in font)

            if is_bold and is_italic:
                text = f"***{text}***"
            elif is_bold:
                text = f"**{text}**"
            elif is_italic:
                text = f"*{text}*"

            spans_text.append(text)
        parts.append("".join(spans_text))
    return "\n".join(parts).strip()


def normalize_paragraph(text: str) -> str:
    """
    Joins internal line breaks, removes simple hyphenation,
    and normalizes spaces.
    """
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    merged = []
    for i, line in enumerate(lines):
        if line.endswith("-") and i < len(lines) - 1:
            # join with next line without the '-'
            next_line = lines[i + 1].lstrip()
            merged.append(line[:-1] + next_line)
            lines[i + 1] = ""
        else:
            merged.append(line)

    joined = " ".join(l for l in merged if l)
    joined = re.sub(r"\s+", " ", joined)
    return joined.strip()


LIST_NUM_RE = re.compile(r"^(\d+[\.\)])\s+")
HEADING_NUMERIC_RE = re.compile(r"^\d+(\.\d+)*\s+")


def is_list_item(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped[0] in ["•", "·", "-", "*"]:
        return True
    if LIST_NUM_RE.match(stripped):
        return True
    return False


def classify_block(text, avg_size, median_size, heading_levels=None, heading_factor=1.2):
    """
    Classifies a block as:
    - 'heading'   → font larger than median / numeric section pattern
    - 'list'      → starts with marker (bullet or numeric)
    - 'paragraph' → normal text
    Returns (type, heading_level_or_None)
    """
    stripped = text.strip()
    if not stripped:
        return "paragraph", None

    # 0) Figure captions don't become headings
    if stripped.lower().startswith("figura "):
        return "paragraph", None

    # 1) List: if majority of lines are list items
    lines = [l for l in stripped.splitlines() if l.strip()]
    if lines and sum(1 for l in lines if is_list_item(l)) >= max(1, len(lines) // 2):
        return "list", None

    # 2) possible numeric heading (1., 1.2, 11.3.4 etc.)
    is_numbered_heading = bool(HEADING_NUMERIC_RE.match(stripped))

    # 3) Formulas / expressions with '=' never become headings
    if "=" in stripped or "≈" in stripped:
        return "paragraph", None

    # 4) try to find heading level by closest font size
    level = None
    if heading_levels:
        closest_size = min(heading_levels.keys(), key=lambda s: abs(s - avg_size))
        if closest_size >= median_size * 1.05:
            level = heading_levels[closest_size]

    # 5) if no match by size, but has numeric section pattern, treat as secondary heading
    if level is None and is_numbered_heading and avg_size >= median_size * 0.95:
        level = 3  # treat as H3

    # 6) Avoid headings that are only numbers (like "6" / "13")
    if level is not None:
        if re.fullmatch(r"\d{1,4}", stripped):
            return "paragraph", None

    if level is not None and len(stripped) > 2:
        return "heading", level

    return "paragraph", None


def maybe_table_block(text: str, min_cols=3, min_rows=3):
    """
    Simple heuristic to detect "tables" in text:
    - multiple lines
    - columns separated by multiple spaces
    Returns list of rows/columns or None.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < min_rows:
        return None

    rows = []
    for line in lines:
        parts = [p.strip() for p in re.split(r"\s{2,}", line.strip()) if p.strip()]
        if len(parts) < min_cols:
            return None
        rows.append(parts)

    cols_counts = {len(r) for r in rows}
    if len(cols_counts) > 2:
        return None

    return rows


def is_probable_page_number_block(
    text: str,
    bbox,
    page_rect,
    median_font_size: float,
    avg_font_size: float,
    top_margin: float = 40,
    bottom_margin: float = 40,
) -> bool:
    """
    Heuristic to detect blocks that are probably page numbers
    (and not formulas):

    - text with only 1–4 digits
    - near top or bottom
    - small font (<= median or slightly smaller)
    - small width
    """
    stripped = text.strip()

    # must be only digits (page number like "6", "13", "120")
    if not re.fullmatch(r"\d{1,4}", stripped):
        return False

    x0, y0, x1, y1 = bbox
    page_h = page_rect.height
    page_w = page_rect.width

    # position: very close to top or bottom
    near_top = y0 < top_margin
    near_bottom = (page_h - y1) < bottom_margin
    if not (near_top or near_bottom):
        return False

    # font: cannot be much larger than median
    if avg_font_size > median_font_size * 1.15:
        return False

    # small width (not a long heading)
    width = x1 - x0
    if width > page_w * 0.25:
        return False

    return True


# =====================================================================
#  Column detection, ordering and block merging
# =====================================================================

def detect_columns(blocks, threshold=50):
    """
    Detects 1 or 2 columns based on x0 coordinate.
    Returns list of columns: [[left_col], [right_col]] or [[all]]
    """
    if not blocks:
        return [[]]

    xs = [b.get("bbox", (0, 0, 0, 0))[0] for b in blocks]
    xs_sorted = sorted(xs)

    if len(xs_sorted) < 2:
        return [blocks]

    diffs = [xs_sorted[i + 1] - xs_sorted[i] for i in range(len(xs_sorted) - 1)]
    max_diff = max(diffs) if diffs else 0

    if max_diff > threshold:
        idx = diffs.index(max_diff)
        split_value = (xs_sorted[idx] + xs_sorted[idx + 1]) / 2.0

        left_col = [b for b in blocks if b.get("bbox", (0, 0, 0, 0))[0] <= split_value]
        right_col = [b for b in blocks if b.get("bbox", (0, 0, 0, 0))[0] > split_value]

        return [left_col, right_col]

    return [blocks]


def extract_page_blocks_ordered(page):
    """
    Returns blocks from a page in reading order.
    - Detects columns
    - Sorts by vertical position (y0)
    """
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])

    # Text blocks only
    text_blocks = [b for b in blocks if b.get("type") == 0]
    if not text_blocks:
        return []

    # Sort by y to stabilize
    text_blocks = sorted(
        text_blocks,
        key=lambda b: (b.get("bbox", (0, 0, 0, 0))[1], b.get("bbox", (0, 0, 0, 0))[0])
    )

    # Detect columns
    columns = detect_columns(text_blocks)

    # Sort within each column
    for col in columns:
        col.sort(key=lambda b: b.get("bbox", (0, 0, 0, 0))[1])

    # Concatenate left → right
    return [blk for col in columns for blk in col]


def merge_similar_blocks(blocks, y_threshold=5, font_tol=0.3):
    """
    Joins consecutive blocks that are probably the same paragraph:
    - same column (similar x0)
    - similar average font
    - small vertical distance
    """
    merged = []
    current = None

    for b in blocks:
        if b.get("type") != 0:
            if current is not None:
                merged.append(current)
                current = None
            merged.append(b)
            continue

        if current is None:
            current = b
            continue

        bx0, by0, bx1, by1 = b.get("bbox", (0, 0, 0, 0))
        cx0, cy0, cx1, cy1 = current.get("bbox", (0, 0, 0, 0))

        b_size = block_average_font_size(b)
        c_size = block_average_font_size(current)

        same_col = abs(bx0 - cx0) < 5
        close_y = (by0 - cy1) >= 0 and (by0 - cy1) < y_threshold
        similar_font = abs(b_size - c_size) <= font_tol

        if same_col and close_y and similar_font:
            current_lines = current.get("lines", [])
            current_lines.extend(b.get("lines", []))
            current["lines"] = current_lines

            bbox = current.get("bbox", (0, 0, 0, 0))
            bbox_list = list(bbox)
            bbox_list[3] = max(cy1, by1)
            current["bbox"] = tuple(bbox_list)
        else:
            merged.append(current)
            current = b

    if current is not None:
        merged.append(current)

    return merged


# =====================================================================
#  Repeated header/footer detection
# =====================================================================

def collect_header_footer_candidates(doc, top_margin=80, bottom_margin=80):
    """
    Collects texts that appear frequently at top/bottom
    to be removed (header/footer).
    """
    header_texts = Counter()
    footer_texts = Counter()

    for page in doc:
        page_dict = page.get_text("dict")
        height = page.rect.height
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            bbox = block.get("bbox", (0, 0, 0, 0))
            y0, y1 = bbox[1], bbox[3]
            text = block_text(block)
            text = text.strip()
            if not text:
                continue

            if y0 < top_margin:
                header_texts[text] += 1
            elif height - y1 < bottom_margin:
                footer_texts[text] += 1

    return header_texts, footer_texts


# =====================================================================
#  Image extraction
# =====================================================================

def calculate_group_bbox(drawings):
    """Calculates the bounding box that encompasses all drawings in a group."""
    if not drawings:
        return (0, 0, 0, 0)

    min_x = min(d['rect'].x0 for d in drawings)
    min_y = min(d['rect'].y0 for d in drawings)
    max_x = max(d['rect'].x1 for d in drawings)
    max_y = max(d['rect'].y1 for d in drawings)

    return (min_x, min_y, max_x, max_y)


def group_drawings_by_proximity(drawings, threshold=50):
    """
    Groups nearby drawings to create composite images.
    Uses smarter grouping that detects separate clusters.
    """
    if not drawings:
        return []

    drawing_items = []
    for d in drawings:
        rect = d.get('rect', fitz.Rect())
        drawing_items.append({
            'rect': rect,
            'data': d
        })

    groups = []
    used = set()

    for i, item in enumerate(drawing_items):
        if i in used:
            continue

        group = [item]
        used.add(i)
        group_bbox = item['rect']

        for j, other_item in enumerate(drawing_items):
            if j in used:
                continue

            rect2 = other_item['rect']

            dx = max(0, group_bbox.x0 - rect2.x1, rect2.x0 - group_bbox.x1)
            dy = max(0, group_bbox.y0 - rect2.y1, rect2.y0 - group_bbox.y1)
            distance = (dx**2 + dy**2) ** 0.5

            if distance < threshold:
                temp_bbox = fitz.Rect(
                    min(group_bbox.x0, rect2.x0),
                    min(group_bbox.y0, rect2.y0),
                    max(group_bbox.x1, rect2.x1),
                    max(group_bbox.y1, rect2.y1)
                )

                width = temp_bbox.x1 - temp_bbox.x0
                height = temp_bbox.y1 - temp_bbox.y0

                if width <= 350 and height <= 350:
                    group.append(other_item)
                    used.add(j)
                    group_bbox = temp_bbox

        groups.append(group)

    return groups


def extract_images_from_page(page, page_num, doc_id: str):
    """
    Extracts all image instances from a page and converts to base64.
    Captures:
    1. Embedded raster images (JPEG, PNG)
    2. Vector graphics and drawings (rendered as images)
    """
    images_info = []
    image_list = page.get_images(full=True)

    xref_cache = {}

    # 1. Embedded raster images
    for img_index, img in enumerate(image_list):
        xref = img[0]

        try:
            if xref not in xref_cache:
                base_image = page.parent.extract_image(xref)
                xref_cache[xref] = {
                    'bytes': base_image["image"],
                    'ext': base_image["ext"]
                }

            image_bytes = xref_cache[xref]['bytes']
            image_ext = xref_cache[xref]['ext']

            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            img_rects = page.get_image_rects(xref)

            if not img_rects:
                image_id = f"{doc_id}_page{page_num + 1}_img{img_index + 1}"
                images_info.append({
                    'id': image_id,
                    'image_data': image_base64,
                    'image_format': image_ext,
                    'page': page_num + 1,
                    'bbox': None,
                    'xref': xref,
                    'size_bytes': len(image_bytes),
                    'type': 'raster'
                })
                logger.debug(
                    f"Extracted raster image: {image_id} ({len(image_bytes)} bytes, format: {image_ext})"
                )
            else:
                for rect_idx, bbox in enumerate(img_rects):
                    image_id = f"{doc_id}_page{page_num + 1}_img{img_index + 1}_inst{rect_idx + 1}"

                    images_info.append({
                        'id': image_id,
                        'image_data': image_base64,
                        'image_format': image_ext,
                        'page': page_num + 1,
                        'bbox': str(bbox) if bbox else None,
                        'xref': xref,
                        'size_bytes': len(image_bytes),
                        'instance': rect_idx + 1,
                        'total_instances': len(img_rects),
                        'type': 'raster'
                    })

                    logger.debug(
                        f"Extracted raster image: {image_id} ({len(image_bytes)} bytes, "
                        f"format: {image_ext}, instance {rect_idx + 1}/{len(img_rects)})"
                    )

        except Exception as e:
            logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
            continue

    # 2. Vector graphics
    try:
        drawings = page.get_drawings()
        if drawings and len(drawings) > 0:
            logger.info(f"Page {page_num + 1}: detected {len(drawings)} vector drawings")

            drawing_groups = group_drawings_by_proximity(drawings, threshold=50)

            for group_idx, drawing_group in enumerate(drawing_groups):
                try:
                    group_bbox = calculate_group_bbox(drawing_group)

                    width = group_bbox[2] - group_bbox[0]
                    height = group_bbox[3] - group_bbox[1]
                    area = width * height

                    MIN_WIDTH = 80
                    MIN_HEIGHT = 80
                    MIN_AREA = 6000

                    aspect_ratio = max(width, height) / (min(width, height) + 0.1)
                    MAX_ASPECT_RATIO = 10

                    MIN_DRAWINGS = 5

                    if width < MIN_WIDTH or height < MIN_HEIGHT:
                        logger.debug(
                            f"Ignoring group {group_idx + 1}: too small ({width:.1f}x{height:.1f})"
                        )
                        continue

                    if area < MIN_AREA:
                        logger.debug(
                            f"Ignoring group {group_idx + 1}: area too small ({area:.0f} px²)"
                        )
                        continue

                    if aspect_ratio > MAX_ASPECT_RATIO:
                        logger.debug(
                            f"Ignoring group {group_idx + 1}: too elongated (ratio {aspect_ratio:.1f})"
                        )
                        continue

                    if len(drawing_group) < MIN_DRAWINGS and area < 10000:
                        logger.debug(
                            f"Ignoring group {group_idx + 1}: few drawings ({len(drawing_group)}) "
                            f"and small area"
                        )
                        continue

                    clip = fitz.Rect(group_bbox)
                    pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2))

                    img_bytes = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    image_id = f"{doc_id}_page{page_num + 1}_vector{group_idx + 1}"
                    images_info.append({
                        'id': image_id,
                        'image_data': img_base64,
                        'image_format': 'png',
                        'page': page_num + 1,
                        'bbox': str(group_bbox),
                        'xref': None,
                        'size_bytes': len(img_bytes),
                        'type': 'vector',
                        'drawings_count': len(drawing_group)
                    })

                    logger.debug(
                        f"Extracted vector graphic: {image_id} ({width:.0f}x{height:.0f}px, "
                        f"{len(img_bytes)} bytes, {len(drawing_group)} drawings)"
                    )

                except Exception as e:
                    logger.warning(
                        f"Error rendering drawing group {group_idx + 1} from page {page_num + 1}: {e}"
                    )

    except Exception as e:
        logger.warning(f"Error processing vector drawings from page {page_num + 1}: {e}")

    return images_info


# =====================================================================
#  Main PDF → Markdown conversion
# =====================================================================

def pdf_to_markdown_fast(pdf_path: Path, doc_id: str = "doc") -> tuple:
    """
    Converts PDF to markdown using PyMuPDF (fast) and extracts images in base64.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document ID

    Returns:
        Tuple: (markdown_string, images_info_list)
    """
    doc = fitz.open(pdf_path)

    # 1st pass: collect font sizes
    font_sizes = collect_font_sizes(doc)
    median_size = statistics.median(font_sizes) if font_sizes else 10.0
    heading_levels = build_heading_levels(font_sizes, max_levels=3)

    # Header/footer detection
    header_texts, footer_texts = collect_header_footer_candidates(doc)
    min_pages = max(3, int(doc.page_count * 0.5))  # appears in at least 50% of pages

    header_trash = {t for t, c in header_texts.items() if c >= min_pages}
    footer_trash = {t for t, c in footer_texts.items() if c >= min_pages}

    # 2nd pass: extract images and generate markdown
    all_images = []
    md_lines = []
    md_lines.append(f"<!-- Generated from {pdf_path.name} using FastPDF -->")
    md_lines.append(
        f"<!-- Pages: {doc.page_count}, median font: {median_size:.2f} -->"
    )
    md_lines.append("<!-- Images stored in database -->")
    md_lines.append("")

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        md_lines.append(f"\n\n---\n\n")
        md_lines.append(f"<!-- Page {page_num + 1} -->\n")

        page_images = extract_images_from_page(page, page_num, doc_id)
        all_images.extend(page_images)

        if page_images:
            if KEEP_DETAILED_IMAGE_LIST:
                md_lines.append(f"\n**Images on page {page_num + 1}:**\n")
                for img in page_images:
                    img_type = img.get('type', 'image')
                    md_lines.append(f"- {img_type}: `{img['id']}`")
                md_lines.append("")
            else:
                total = len(page_images)
                raster = sum(1 for img in page_images if img.get("type") == "raster")
                vector = sum(1 for img in page_images if img.get("type") == "vector")
                md_lines.append(
                    f"\n<!-- Images on page {page_num + 1}: {total} "
                    f"(raster={raster}, vector={vector}) -->\n"
                )

        blocks = extract_page_blocks_ordered(page)
        blocks = merge_similar_blocks(blocks)

        for block in blocks:
            if block.get("type") != 0:
                continue

            text = block_text(block)
            text = text.strip()
            if not text:
                continue

            bbox = block.get("bbox", (0, 0, 0, 0))
            avg_size = block_average_font_size(block)

            if text in header_trash or text in footer_trash:
                continue

            if is_probable_page_number_block(
                text=text,
                bbox=bbox,
                page_rect=page.rect,
                median_font_size=median_size,
                avg_font_size=avg_size,
            ):
                continue

            # Tables (simple heuristic)
            table = maybe_table_block(text)
            if table:
                header = table[0]
                md_lines.append("| " + " | ".join(header) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for row in table[1:]:
                    md_lines.append("| " + " | ".join(row) + " |")
                md_lines.append("")
                continue

            kind, level = classify_block(text, avg_size, median_size, heading_levels)

            if kind == "heading":
                level = min(level or 2, 6)
                hashes = "#" * level
                md_lines.append(f"{hashes} {text.strip()}\n")

            elif kind == "list":
                for line in text.splitlines():
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if line_stripped[0] in ["•", "·", "-", "*"]:
                        line_stripped = line_stripped[1:].strip()
                    # keep numbering if exists (1. / 1) etc.
                    md_lines.append(f"- {line_stripped}")
                md_lines.append("")

            else:
                paragraph = normalize_paragraph(text)
                if paragraph:
                    md_lines.append(paragraph)
                    md_lines.append("")

    total_pages = doc.page_count
    doc.close()

    logger.info(f"FastPDF extracted {len(all_images)} images from {total_pages} pages")

    return "\n".join(md_lines), all_images


def process_pdf_fast(pdf_path: str, output_path: str = None, doc_id: str = "doc") -> Dict[str, any]:
    """
    Processes a PDF using FastPDF, extracts images in base64 and optionally saves the markdown.

    Args:
        pdf_path: Input PDF path
        output_path: Output .md file path (optional)
        doc_id: Document ID

    Returns:
        Dict with statistics: {
            'success': bool,
            'elapsed_time': float,
            'markdown': str,
            'markdown_size': int,
            'images_count': int,
            'images_info': list,
            'error': str (if any)
        }
    """
    try:
        start_time = time.time()

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return {
                'success': False,
                'error': f'File not found: {pdf_path}'
            }

        logger.info(f"Starting FastPDF conversion: {pdf_file.name}")

        markdown, images_info = pdf_to_markdown_fast(pdf_file, doc_id)

        # Save markdown to file only if output_path is provided
        if output_path:
            output_file = Path(output_path)
            output_file.write_text(markdown, encoding='utf-8')
            logger.info(f"Markdown saved at: {output_path}")

        elapsed = time.time() - start_time

        logger.info(f"FastPDF completed: {pdf_file.name} in {elapsed:.2f}s")
        logger.info(
            f"  Markdown: {len(markdown):,} characters, {len(markdown.splitlines()):,} lines"
        )
        logger.info(f"  Images: {len(images_info)} extracted in base64")

        return {
            'success': True,
            'elapsed_time': elapsed,
            'markdown': markdown,
            'markdown_size': len(markdown),
            'markdown_lines': len(markdown.splitlines()),
            'images_count': len(images_info),
            'images_info': images_info,
            'output_path': str(output_path) if output_path else None
        }

    except Exception as e:
        logger.error(f"Error in FastPDF: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def process_pdf_fast_async(pdf_path: str, output_path: str = None, doc_id: str = "doc") -> Dict[str, any]:
    """
    Asynchronous version of process_pdf_fast.
    Executes processing in separate thread to avoid blocking the event loop.

    Args:
        pdf_path: Input PDF path
        output_path: Output .md file path (optional)
        doc_id: Document ID

    Returns:
        Dict with processing result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        process_pdf_fast,
        pdf_path,
        output_path,
        doc_id
    )
