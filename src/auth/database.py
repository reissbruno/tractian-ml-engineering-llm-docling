import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.logger import logger

SQLALCHEMY_DATABASE_URL = "sqlite:///./tractian.db"
DATABASE_FILE = "tractian.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=True)  # Kept for compatibility, but no longer used
    pdf_data = Column(Text, nullable=True)  # PDF in base64
    file_size = Column(Integer)
    status = Column(String, default="pending")
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)


class DocumentImage(Base):
    __tablename__ = "document_images"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_data = Column(Text, nullable=False)  # Base64 string
    image_format = Column(String, default="png")
    caption = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """
    Initialize the database, creating tables if necessary.
    Checks if the tractian.db file already exists and logs the information.
    """
    db_exists = os.path.exists(DATABASE_FILE)

    if db_exists:
        logger.info(f"Database '{DATABASE_FILE}' already exists.")
    else:
        logger.info(f"Database '{DATABASE_FILE}' not found. Creating new database...")

    # Create tables (does nothing if they already exist)
    Base.metadata.create_all(bind=engine)

    if not db_exists:
        logger.info(f"Database '{DATABASE_FILE}' created successfully.")


def get_db():
    """
    Dependency to get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
