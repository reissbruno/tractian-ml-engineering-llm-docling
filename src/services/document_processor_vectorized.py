"""
Service for document processing with vectorization.
Processes PDFs, generates chunks and indexes in vector store.
"""

import time
from datetime import datetime
from typing import Dict, List

from src.logger import logger
from src.auth.database import SessionLocal, Document, DocumentImage
from src.services.chunking_service import create_chunker
from src.services.content_cleaner import clean_markdown_content, split_by_pages
from src.services.document_analyzer import get_document_analyzer
from src.services.document_processor import process_document_to_markdown_parallel
from src.services.embeddings_service import get_embeddings_service
from src.services.fast_pdf_processor import process_pdf_fast_async
from src.services.vector_store_service import get_vector_store_service

# Processing status
processing_status: Dict[str, Dict] = {}


async def process_single_document(
    file_path: str,
    doc_id: str,
    user_id: int,
    filename: str,
    processor: str = "fast"
) -> Dict:
    """
    Processes a single PDF document.

    Args:
        file_path: PDF file path
        doc_id: Document ID
        user_id: User ID
        filename: File name
        processor: Processor type ('fast' or 'docling')

    Returns:
        Dict with processing result
    """
    start_time = time.time()

    try:
        logger.info(f"Starting processing: {filename} (processor={processor})")

        # Update status
        processing_status[doc_id] = {
            "status": "processing",
            "progress": 10,
            "message": "Extracting text and images...",
            "filename": filename
        }

        # 1. Process PDF → Markdown + Images (execute asynchronously)
        if processor == "fast":
            result = await process_pdf_fast_async(file_path, output_path=None, doc_id=doc_id)
        else:
            markdown_text = await process_document_to_markdown_parallel(file_path, doc_id)

            result_images = await process_pdf_fast_async(file_path, output_path=None, doc_id=doc_id)

            if not result_images['success']:
                raise Exception(result_images.get('error', 'Error extracting images'))

            result = {
                'success': True,
                'markdown': markdown_text,
                'images_info': result_images.get('images_info', [])
            }

        if not result['success']:
            raise Exception(result.get('error', 'Unknown error in processing'))

        markdown_text = result.get('markdown', '')
        images_info = result.get('images_info', [])

        logger.info(f"Markdown generated: {len(markdown_text)} characters, {len(images_info)} images")

        if processor == "docling":
            debug_path = f"debug_markdown_{doc_id}.md"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"DEBUG: Docling markdown saved to {debug_path}")

        # 2. Clean content (only for FastPDF, Docling comes clean)
        if processor == "fast":
            processing_status[doc_id].update({
                "progress": 25,
                "message": "Cleaning content..."
            })

            cleaned_markdown = clean_markdown_content(markdown_text)
            logger.info(f"Content cleaned: {len(cleaned_markdown)} characters")
        else:
            cleaned_markdown = markdown_text
            logger.info(f"Using Docling markdown: {len(cleaned_markdown)} characters")

        processing_status[doc_id].update({
            "progress": 35,
            "message": "Splitting by pages..."
        })

        # 3. Split by pages
        pages = split_by_pages(cleaned_markdown)
        logger.info(f"Split into {len(pages)} pages")

        if len(pages) == 0:
            logger.error(f"ERROR: No pages detected after split!")
            logger.error(f"Cleaned markdown size: {len(cleaned_markdown)}")
            logger.error(f"Markdown preview (first 500 chars): {cleaned_markdown[:500]}")
        else:
            logger.info(f"First page has {len(pages[0][1])} characters")

        processing_status[doc_id].update({
            "progress": 45,
            "message": "Analyzing document structure..."
        })

        # 4. Analyze document structure and determine chunking configuration
        analyzer = get_document_analyzer()
        chunking_config = analyzer.get_chunking_config(
            text=cleaned_markdown,
            num_pages=len(pages)
        )

        processing_status[doc_id].update({
            "progress": 50,
            "message": f"Creating chunks ({chunking_config['strategy']})..."
        })

        # 5. Create chunks with adaptive configuration
        chunker = create_chunker(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap']
        )

        base_metadata = {
            'doc_id': doc_id,
            'doc_name': filename,
            'parser': processor,
            'language': 'pt-BR',
            'version': 1,
            'total_pages': len(pages),
            'chunking_strategy': chunking_config['strategy'],
            'chunk_size': chunking_config['chunk_size']
        }

        chunks = chunker.chunk_by_pages(pages, base_metadata)
        chunks = chunker.enrich_chunks_with_content_type(chunks)

        logger.info(
            f"Created {len(chunks)} chunks "
            f"[{chunking_config['strategy']}: {chunking_config['chunk_size']} chars, "
            f"{chunking_config['chunk_overlap']} overlap]"
        )

        processing_status[doc_id].update({
            "progress": 60,
            "message": "Relating images to chunks..."
        })

        # 5. Relate images to chunks
        chunks = enrich_chunks_with_images(chunks, images_info)

        processing_status[doc_id].update({
            "progress": 70,
            "message": "Generating embeddings..."
        })

        # 6. Generate embeddings (asynchronous to not block)
        embeddings_service = get_embeddings_service()
        texts = [chunk['content'] for chunk in chunks]
        embeddings = await embeddings_service.encode_async(texts)

        logger.info(f"Embeddings generated: {len(embeddings)} vectors")

        processing_status[doc_id].update({
            "progress": 85,
            "message": "Indexing in vector store..."
        })

        # 7. Add to vector store
        vector_store = get_vector_store_service()
        chunks_added = vector_store.add_chunks(user_id, chunks, embeddings)

        logger.info(f"Chunks indexed: {chunks_added}")

        processing_status[doc_id].update({
            "progress": 95,
            "message": "Saving images to database..."
        })

        # 8. Save images to database
        images_saved = save_images_to_database(doc_id, images_info)

        logger.info(f"Images saved: {images_saved}")

        # 9. Update document in database
        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "completed"
                doc.chunks_count = chunks_added
                doc.processed_at = datetime.utcnow()
                db.commit()
                logger.info(f"Document updated in database: {doc_id}")
        finally:
            db.close()

        elapsed = time.time() - start_time

        processing_status[doc_id].update({
            "progress": 100,
            "message": "Processing completed!",
            "status": "completed"
        })

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": chunks_added,
            "images_count": images_saved,
            "elapsed_time": elapsed
        }

    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}", exc_info=True)

        # Update error status
        processing_status[doc_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Error: {str(e)}",
            "filename": filename
        }

        # Update in database
        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "failed"
                doc.error_message = str(e)
                db.commit()
        finally:
            db.close()

        return {
            "success": False,
            "doc_id": doc_id,
            "filename": filename,
            "error": str(e)
        }


def enrich_chunks_with_images(chunks: List[Dict], images_info: List[Dict]) -> List[Dict]:
    """
    Relates images to chunks based on page number.

    Args:
        chunks: List of chunks
        images_info: List of image information

    Returns:
        Chunks enriched with image metadata
    """
    page_to_images = {}
    for img in images_info:
        page = img.get('page')
        if page not in page_to_images:
            page_to_images[page] = []
        page_to_images[page].append(img['id'])

    for chunk in chunks:
        page = chunk['metadata'].get('page_number')
        if page and page in page_to_images:
            chunk['metadata']['has_images'] = 'true'
            chunk['metadata']['image_ids'] = ','.join(page_to_images[page])
        else:
            chunk['metadata']['has_images'] = 'false'
            chunk['metadata']['image_ids'] = ''

    return chunks


def save_images_to_database(doc_id: str, images_info: List[Dict]) -> int:
    """
    Saves images to database.

    Args:
        doc_id: Document ID
        images_info: List of image information

    Returns:
        Number of images saved
    """
    if not images_info:
        return 0

    db = SessionLocal()
    try:
        for img_info in images_info:
            db_image = DocumentImage(
                id=img_info['id'],
                document_id=doc_id,
                page_number=img_info['page'],
                image_data=img_info['image_data'],
                image_format=img_info['image_format'],
                caption=f"Image from page {img_info['page']}"
            )
            db.add(db_image)

        db.commit()
        logger.info(f"{len(images_info)} images saved to database")
        return len(images_info)

    except Exception as e:
        logger.error(f"Error saving images to database: {e}")
        db.rollback()
        return 0

    finally:
        db.close()


async def process_documents_batch(
    files_info: List[Dict],
    user_id: int,
    processor: str = "fast"
) -> Dict:
    """
    Processes multiple documents in batch.

    Args:
        files_info: List of dicts with file info
        user_id: User ID
        processor: Processor type

    Returns:
        Dict with batch processing result
    """
    results = []
    total_chunks = 0
    total_images = 0

    for file_info in files_info:
        result = await process_single_document(
            file_path=file_info['file_path'],
            doc_id=file_info['doc_id'],
            user_id=user_id,
            filename=file_info['filename'],
            processor=processor
        )

        results.append(result)

        if result['success']:
            total_chunks += result.get('chunks_count', 0)
            total_images += result.get('images_count', 0)

    successful = sum(1 for r in results if r['success'])

    return {
        "documents_indexed": successful,
        "total_chunks": total_chunks,
        "total_images": total_images,
        "results": results
    }


def get_processing_status(doc_id: str) -> Dict:
    """
    Returns the processing status of a document.

    Args:
        doc_id: Document ID

    Returns:
        Dict with status
    """
    return processing_status.get(doc_id, {
        "status": "unknown",
        "progress": 0,
        "message": "Unknown status"
    })
