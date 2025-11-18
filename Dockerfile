# ============================================
# STAGE 1: Builder - Build dependencies
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (only what's needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies in user directory (cacheable)
RUN pip install --user --no-cache-dir --compile -r requirements.txt

# ============================================
# STAGE 2: Runtime - Optimized final image
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install ONLY runtime dependencies (much lighter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy ONLY necessary files (reduces context significantly)
COPY server.py .
COPY src/ ./src/
COPY static/ ./static/
COPY .env* ./

# Create necessary directories with appropriate permissions
RUN mkdir -p chroma_db static/uploads logs && \
    chmod -R 755 chroma_db logs

# Expose port
EXPOSE 8000

# Health check (basic server check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/ || exit 1

# Run the application with uvicorn
# Using 1 worker for development, increase for production
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]