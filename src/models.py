from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question")


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Answer")
    references: List[str] = Field(..., description="References")


class RegisterRequest(BaseModel):
    user_name: str = Field(..., description="User Name")
    senha: str = Field(..., description="Senha")


class LoginRequest(BaseModel):
    user_name: str = Field(..., description="User Name")
    senha: str = Field(..., description="Senha")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Access Token")
    token_type: str = Field(..., description="Token Type")


class DocumentInfo(BaseModel):
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Filename")
    status: str = Field(..., description="Status")
    chunks_count: Optional[int] = Field(None, description="Chunks Count")
    file_size: Optional[int] = Field(None, description="File Size in bytes")
    created_at: Optional[str] = Field(None, description="Created At")


class DocumentsListResponse(BaseModel):
    documents: List[DocumentInfo] = Field(..., description="Documents")


class UploadResponse(BaseModel):
    message: str = Field(..., description="Message")
    documents_indexed: int = Field(..., description="Documents Indexed")
    total_chunks: int = Field(..., description="Total Chunks")


class UploadAsyncResponse(BaseModel):
    message: str = Field(..., description="Message")
    doc_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Status")


class DeleteResponse(BaseModel):
    message: str = Field(..., description="Message")
    doc_id: str = Field(..., description="Document ID")
