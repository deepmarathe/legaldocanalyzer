# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Tesseract OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract languages for multilingual OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    tesseract-ocr-tel \
    tesseract-ocr-mar \
    tesseract-ocr-ben \
    tesseract-ocr-guj \
    tesseract-ocr-kan \
    tesseract-ocr-mal \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Set environment variable for NLTK data location
ENV NLTK_DATA=/root/nltk_data

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install necessary NLTK resources explicitly
RUN python -c "import nltk; \
    nltk.data.path.append('/root/nltk_data'); \
    nltk.download('punkt', download_dir='/root/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/root/nltk_data'); \
    nltk.download('stopwords', download_dir='/root/nltk_data')"

# Install SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables for better Docker compatibility
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
