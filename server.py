import os
import uuid
import base64
import tempfile
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.models import (
    QuestionRequest,
    QuestionResponse,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    DocumentInfo,
    DocumentsListResponse
)

from src.logger import logger
from src.auth.database import get_db, init_db, User, Document, DocumentImage
from src.auth.auth import hash_password, verify_password, create_access_token
from src.services.document_processor import (
    process_document_background,
    get_processing_status
)
from src.services.document_processor_vectorized import process_documents_batch
from src.services.embeddings_service import get_embeddings_service
from src.services.vector_store_service import get_vector_store_service
from src.services.llm_service import get_llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Tractian RAG application...")
    init_db()
    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down Tractian RAG application...")


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/login.html")


@app.post("/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.user_name == request.user_name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = hash_password(request.senha)
    new_user = User(user_name=request.user_name, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully", "user_id": new_user.id}


@app.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_name == request.user_name).first()
    if not user or not verify_password(request.senha, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user.user_name})
    return TokenResponse(access_token=access_token, token_type="bearer")


@app.get("/documents", response_model=DocumentsListResponse)
async def list_documents(db: Session = Depends(get_db)):
    current_user = User(id=1, user_name="test")
    docs = db.query(Document).filter(Document.user_id == current_user.id).all()

    documents = [
        DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            status=doc.status,
            chunks_count=doc.chunks_count,
            file_size=doc.file_size,
            created_at=doc.created_at.isoformat() if doc.created_at else None
        )
        for doc in docs
    ]

    return DocumentsListResponse(documents=documents)


@app.post("/documents")
async def upload_documents_vectorized(
    files: List[UploadFile] = File(...),
    processor: str = Form("fast"),
    db: Session = Depends(get_db)
):
    current_user = User(id=1, user_name="test")
    logger.info(f"Uploading {len(files)} file(s) | processor={processor}")

    for file in files:
        if file.content_type != 'application/pdf':
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a PDF (type: {file.content_type})"
            )

    files_info = []
    temp_files = []

    for file in files:
        doc_id = str(uuid.uuid4())
        content = await file.read()
        pdf_base64 = base64.b64encode(content).decode('utf-8')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb')
        temp_file.write(content)
        temp_file.close()
        temp_files.append(temp_file.name)

        doc = Document(
            id=doc_id,
            user_id=current_user.id,
            filename=file.filename,
            file_path=None,
            pdf_data=pdf_base64,
            file_size=len(content),
            status="processing",
            created_at=datetime.utcnow()
        )
        db.add(doc)

        files_info.append({
            'doc_id': doc_id,
            'filename': file.filename,
            'file_path': temp_file.name
        })

    db.commit()
    logger.info(f"{len(files)} files saved to database (base64), starting processing...")

    try:
        result = await process_documents_batch(
            files_info=files_info,
            user_id=current_user.id,
            processor=processor
        )

        logger.info(
            f"Batch completed: {result['documents_indexed']} docs, "
            f"{result['total_chunks']} chunks, {result['total_images']} images"
        )

        return {
            "message": "Documents processed successfully",
            "documents_indexed": result['documents_indexed'],
            "total_chunks": result['total_chunks']
        }

    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Arquivo temporario removido: {temp_path}")
            except Exception as e:
                logger.warning(f"Nao foi possivel remover arquivo temporario {temp_path}: {e}")


@app.post("/documents/upload")
async def upload_document_async(
    file: UploadFile = File(...),
    processor: str = Form("fast"),
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(id=1, user_name="test"))
):
    logger.info(f"Processor received: '{processor}' | File: {file.filename}")

    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

    doc_id = str(uuid.uuid4())
    logger.info(f"Processing file: {file.filename}")
    content = await file.read()

    pdf_base64 = base64.b64encode(content).decode('utf-8')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb')
    temp_file.write(content)
    temp_file.close()
    file_path = temp_file.name

    doc = Document(
        id=doc_id,
        user_id=current_user.id,
        filename=file.filename,
        file_path=None,
        pdf_data=pdf_base64,
        file_size=len(content),
        status="processing",
        created_at=datetime.utcnow()
    )
    db.add(doc)
    db.commit()

    if processor not in ['docling', 'fast']:
        processor = 'fast'

    asyncio.create_task(process_document_background(
        file_path=file_path,
        doc_id=doc_id,
        user_id=current_user.id,
        processor=processor
    ))

    return {
        "message": "Document received and processing started",
        "doc_id": doc_id,
        "status": "processing"
    }


@app.get("/documents/{doc_id}/progress")
async def track_document_progress(doc_id: str):
    async def generate():
        while True:
            status = get_processing_status(doc_id)

            yield f"data: {status['status']}|{status['progress']}|{status['message']}\n\n"

            if status['status'] in ['completed', 'error']:
                break

            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/documents/{doc_id}/status")
async def get_document_status_endpoint(doc_id: str):
    status = get_processing_status(doc_id)
    return status


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(id=1, user_name="test"))
):
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)
            logger.info(f"PDF file deleted: {doc.file_path}")
    except Exception as e:
        logger.warning(f"Error deleting PDF file: {e}")

    try:
        markdown_dir = f"output_markdowns/user_{current_user.id}"
        markdown_filename = os.path.splitext(doc.filename)[0] + ".md"
        markdown_path = os.path.join(markdown_dir, markdown_filename)

        if os.path.exists(markdown_path):
            os.remove(markdown_path)
            logger.info(f"Markdown deleted: {markdown_path}")
    except Exception as e:
        logger.warning(f"Error deleting markdown: {e}")

    try:
        images = db.query(DocumentImage).filter(DocumentImage.document_id == doc_id).all()
        images_count = len(images)

        if images_count > 0:
            for img in images:
                db.delete(img)
            logger.info(f"{images_count} images deleted from database")
    except Exception as e:
        logger.warning(f"Error deleting images from database: {e}")

    try:
        vector_store = get_vector_store_service()
        collection = vector_store.get_or_create_collection(current_user.id)

        results = collection.get(where={"doc_id": doc_id}, include=[])

        if results and results['ids']:
            chunks_count = len(results['ids'])
            collection.delete(ids=results['ids'])
            logger.info(f"{chunks_count} chunks deleted from ChromaDB (vector store)")
        else:
            logger.info(f"No chunks found in ChromaDB for document {doc_id}")
    except Exception as e:
        logger.warning(f"Error deleting chunks from ChromaDB: {e}")

    db.delete(doc)
    db.commit()

    logger.info(f"Document completely deleted: {doc_id}")

    return {
        "message": "Document deleted successfully (including images and vector store chunks)",
        "doc_id": doc_id
    }


@app.post("/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(id=1, user_name="test")),
    x_llm_provider: str = Header(None, alias="X-LLM-Provider"),
    x_llm_model: str = Header(None, alias="X-LLM-Model")
):
    logger.info(f"Query recebida do usuario {current_user.id}: {request.question}")
    logger.info(f"Provider: {x_llm_provider}, Model: {x_llm_model}")

    try:
        embeddings_service = get_embeddings_service()
        query_embedding = embeddings_service.encode_single(request.question)

        logger.info(f"Embedding da query gerado: {len(query_embedding)} dimensoes")

        vector_store = get_vector_store_service()
        results = vector_store.search(
            user_id=current_user.id,
            query_embedding=query_embedding,
            n_results=5,
        )

        logger.info(f"Busca retornou {len(results['ids'][0])} resultados")

        references = []
        context_chunks = []
        image_ids = []

        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                context_chunks.append(doc_text)

                doc_name = metadata.get('doc_name', 'Unknown document')
                page_num = metadata.get('page_number', '?')

                ref_with_source = f"[{doc_name} - Page {page_num}]"
                references.append(ref_with_source)

                if metadata.get('has_images') and metadata.get('image_ids'):
                    img_ids_str = metadata.get('image_ids', '')
                    if img_ids_str:
                        image_ids.extend(img_ids_str.split(','))

        if not context_chunks:
            answer = (
                "No relevant information found in the indexed documents about this question. "
                "Try rephrasing the question or verify that the correct documents were uploaded."
            )
            return QuestionResponse(answer=answer, references=[])

        images_base64 = []
        if image_ids:
            unique_image_ids = list(set(image_ids))
            logger.info(f"Fetching {len(unique_image_ids)} unique images from database")
            db_images = db.query(DocumentImage).filter(DocumentImage.id.in_(unique_image_ids)).all()
            images_base64 = [img.image_data for img in db_images if img.image_data]
            logger.info(f"Loaded {len(images_base64)} images for multimodal context")

        llm_provider = x_llm_provider or os.getenv("LLM_PROVIDER", "openai")
        llm_model = x_llm_model or os.getenv("LLM_MODEL")

        logger.info(f"Initializing LLM service: provider={llm_provider}, model={llm_model}")
        llm_service = get_llm_service(provider=llm_provider, model_name=llm_model)

        logger.info(f"Generating answer with {len(context_chunks)} chunks and {len(images_base64)} images")
        answer = llm_service.generate_answer(
            question=request.question,
            context_chunks=context_chunks,
            images=images_base64 if images_base64 else None
        )

        logger.info(f"Answer generated by LLM with {len(references)} references")

        return QuestionResponse(
            answer=answer,
            references=references
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return QuestionResponse(
            answer=f"Error processing question: {str(e)}",
            references=[]
        )
