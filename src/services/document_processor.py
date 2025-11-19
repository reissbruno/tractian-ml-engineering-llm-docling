"""
Document processing service with Docling.
Generates markdown from PDFs using parallel processing with isolated workers (bypass GIL).
"""

import asyncio
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from docling.datamodel.pipeline_options import PdfPipelineOptions

from src.logger import logger


def get_optimal_worker_count() -> int:
    """
    Determines the optimal number of workers based on system resources.
    IMPORTANT: Limited to 2-3 workers to avoid exhausting RAM,
    as each worker loads Docling models (~2-3GB each).

    Returns:
        Optimized number of workers for parallel processing
    """
    try:
        cpu_count = multiprocessing.cpu_count()

        # Conservative strategy to avoid OOM (Out of Memory):
        # Docling loads heavy models (~2-3GB per worker)
        # Use AT MOST 5 workers to avoid exhausting RAM

        # FORCE only 1 worker to avoid Windows memory error
        # OSError 1455: "The paging file is too small"
        # Each worker loads TableFormer (~2-3GB) causing OOM
        optimal = 1

        logger.info(f"CPU cores detected: {cpu_count} | Workers: {optimal} (forced to 1 to avoid OOM on Windows)")

        return optimal
    except Exception as e:
        logger.warning(f"Error detecting CPU cores: {e}")
        return 1  # Safe fallback (1 worker for Windows)


def create_pdf_pipeline_options(use_gpu: bool = False):
    """
    Creates optimized configurations for the Docling pipeline.
    """
    

    pipeline = PdfPipelineOptions()

    # OCR and text processing
    pipeline.do_ocr = False  # Digital PDFs (faster)
    pipeline.do_table_structure = True
    pipeline.table_structure_options.do_cell_matching = True

    # Disable unnecessary features for simple markdown
    pipeline.do_picture_classification = False
    pipeline.do_picture_description = False
    pipeline.generate_page_images = False
    pipeline.generate_picture_images = False
    pipeline.generate_table_images = False
    pipeline.generate_parsed_pages = False
    pipeline.do_code_enrichment = False
    pipeline.do_formula_enrichment = False
    pipeline.force_backend_text = False

    # Image scaling (irrelevant with images disabled)
    pipeline.images_scale = 1.0

    # BATCH / PERFORMANCE - OPTIMIZED
    # Larger batch sizes for better throughput
    pipeline.ocr_batch_size = 16
    pipeline.layout_batch_size = 16
    pipeline.table_batch_size = 16

    # Queue and polling control - OPTIMIZED
    pipeline.batch_polling_interval_seconds = 0.05  # Faster polling
    pipeline.queue_max_size = 200  # Larger queue for better throughput

    return pipeline


def split_pdf_into_batches(pdf_path: str, batch_size: int) -> Tuple[List[Tuple[str, int, int]], int]:
    """
    Splits a PDF into batches of pages for parallel processing.

    Args:
        pdf_path: Path to PDF file
        batch_size: Number of pages per batch

    Returns:
        Tuple with (list of batches, total pages)
        Each batch is (pdf_path, start_page, end_page)
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    batches = []
    for start_page in range(0, total_pages, batch_size):
        end_page = min(start_page + batch_size, total_pages)
        batches.append((pdf_path, start_page, end_page))

    return batches, total_pages


def process_pdf_batch(args):
    """
    Processes a batch of PDF pages in parallel.
    IMPORTANT: Standalone function for ProcessPoolExecutor (must be picklable).

    Args:
        args: Tuple with (use_gpu, pdf_path, start_page, end_page, batch_id, temp_dir_str)

    Returns:
        Tuple (batch_id, markdown_content, success, error_msg)
    """
    use_gpu, pdf_path, start_page, end_page, batch_id, temp_dir_str = args
    temp_dir = Path(temp_dir_str)
    temp_pdf = None

    try:
        # Imports inside function (required for multiprocessing)
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        import fitz

        logger.info(f"Batch {batch_id}: extracting pages {start_page+1}-{end_page}")

        temp_pdf = temp_dir / f"batch_{batch_id:03d}_pages_{start_page+1}-{end_page}.pdf"

        try:
            doc = fitz.open(pdf_path)
            new_doc = fitz.open()
            for page_num in range(start_page, end_page):
                if page_num < len(doc):
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_doc.save(str(temp_pdf))
            new_doc.close()
            doc.close()
        except Exception as e:
            return (batch_id, None, False, f"Failed to extract pages: {e}")

        logger.info(f"Batch {batch_id}: processing pages {start_page+1}-{end_page}")

        # Create pipeline options (each process needs its own)
        pipeline = PdfPipelineOptions()
        pipeline.do_ocr = False
        pipeline.do_table_structure = True
        pipeline.table_structure_options.do_cell_matching = True
        pipeline.do_picture_classification = False
        pipeline.do_picture_description = False
        pipeline.generate_page_images = False
        pipeline.generate_picture_images = False
        pipeline.generate_table_images = False
        pipeline.generate_parsed_pages = False
        pipeline.do_code_enrichment = False
        pipeline.do_formula_enrichment = False
        pipeline.ocr_batch_size = 16
        pipeline.layout_batch_size = 16
        pipeline.table_batch_size = 16
        pipeline.batch_polling_interval_seconds = 0.05
        pipeline.queue_max_size = 200

        # Create SEPARATE converter for this process
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
            }
        )

        # Process the temporary PDF
        result = converter.convert(str(temp_pdf))
        markdown = result.document.export_to_markdown()

        # Add page markers to markdown
        # Insert marker at the beginning of each page in the batch
        markdown_with_pages = []
        for page_offset in range(end_page - start_page):
            page_num = start_page + page_offset + 1
            markdown_with_pages.append(f"\n\n---\n\n<!-- Page {page_num} -->\n\n")

        # If batch has only 1 page, add marker at beginning
        # If multiple, split markdown approximately
        if end_page - start_page == 1:
            markdown_with_marker = f"<!-- Page {start_page + 1} -->\n\n{markdown}"
        else:
            # For multiple pages, add marker at beginning only
            # (exact division requires more complex Docling analysis)
            markdown_with_marker = f"<!-- Page {start_page + 1} -->\n\n{markdown}"

        logger.info(f"Batch {batch_id}: completed ({len(markdown_with_marker):,} chars)")

        return (batch_id, markdown_with_marker, True, None)
    except Exception as e:
        import traceback
        error_msg = f"Error in batch {batch_id}: {str(e)}"
        logger.error(f"{error_msg}")
        traceback.print_exc()
        return (batch_id, None, False, error_msg)
    finally:
        # Clean up temporary PDF
        if temp_pdf and temp_pdf.exists():
            try:
                temp_pdf.unlink()
            except:
                pass


async def process_document_to_markdown_parallel(file_path: str, doc_id: str) -> str:
    """
    Processes a PDF using Docling with parallel processing (ProcessPoolExecutor).
    Splits the PDF into batches and processes multiple batches simultaneously.

    Args:
        file_path: Path to PDF file
        doc_id: Document ID for tracking

    Returns:
        String with complete markdown content
    """
    logger.info(f"Processing document {doc_id} with parallel workers...")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix=f"docling_{doc_id}_"))

    try:
        # Analyze PDF to determine configurations
        import fitz
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()

        logger.info(f"PDF with {total_pages} pages")

        # Determine workers and batch size
        max_workers = get_optimal_worker_count()

        # Auto-detect optimized batch_size
        if total_pages < 30:
            batch_size = max(5, total_pages // max_workers)
        elif total_pages < 100:
            batch_size = 8
        else:
            batch_size = max(5, min(10, total_pages // (max_workers * 2)))

        logger.info(f"Batch size: {batch_size} pages | Workers: {max_workers}")

        # Split into batches
        batches, _ = split_pdf_into_batches(file_path, batch_size)
        logger.info(f"Total batches: {len(batches)}")

        # Process batches in parallel using ProcessPoolExecutor
        results = []

        # Execute in separate event loop to avoid blocking
        loop = asyncio.get_event_loop()

        def run_parallel_processing():
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Prepare arguments for all batches
                batch_args = []
                for i, (_, start_page, end_page) in enumerate(batches):
                    batch_args.append((False, file_path, start_page, end_page, i, str(temp_dir)))

                futures = {executor.submit(process_pdf_batch, args): args[4] for args in batch_args}

                batch_results = []
                for future in as_completed(futures):
                    batch_id, markdown, success, error = future.result()

                    if success:
                        batch_results.append((batch_id, markdown))
                        logger.info(f"Batch {batch_id} processed ({len(markdown):,} chars)")
                    else:
                        logger.error(f"Batch {batch_id} failed: {error}")
                        raise Exception(f"Failed to process batch {batch_id}: {error}")

                return batch_results

        results = await loop.run_in_executor(None, run_parallel_processing)

        results.sort(key=lambda x: x[0])
        markdown_full = "\n\n".join([markdown for _, markdown in results])

        logger.info(f"Complete markdown generated: {len(markdown_full):,} characters")

        return markdown_full

    finally:
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


# Configure multiprocessing for Windows
if sys.platform == 'win32':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already configured
