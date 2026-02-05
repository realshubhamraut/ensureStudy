FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including LibreOffice for PPTX→PDF conversion
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    # LibreOffice for PPTX/DOCX to PDF conversion
    libreoffice-writer \
    libreoffice-impress \
    libreoffice-common \
    # Tesseract for OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Fonts for better document rendering
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
