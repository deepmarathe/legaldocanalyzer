# Smart Legal Document Analyzer

A Streamlit-based application for analyzing legal documents with OCR capabilities, document understanding, and legal expertise features.

## Features

- Document text extraction from PDF, images, and DOCX formats
- OCR for scanned documents using Tesseract
- Multi-language support with translation capabilities
- Legal document analysis including:
  - Document type identification
  - Simplified explanations
  - Legal term definitions
  - Red flag identification
  - Action item generation
  - Property detail extraction
  - Stamp duty calculation
  - Enforceability analysis
  - Legal precedent analysis
  - Timeline requirements interpretation

## Docker Setup

### Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of available RAM (OCR operations can be memory-intensive)

### Quick Start

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd smart-legal-document-analyzer
   ```

2. Ensure the file structure is correct:
   ```
   .
   ├── app.py                # Renamed from paste-2.txt
   ├── document_processor.py # Renamed from paste.txt
   ├── Dockerfile
   ├── docker-compose.yml
   └── requirements.txt
   ```

3. Rename the files:
   ```bash
   mv paste.txt document_processor.py
   mv paste-2.txt app.py
   ```

4. Update the import in app.py:
   ```bash
   sed -i 's/from doc1 import EnhancedLegalDocumentProcessor/from document_processor import EnhancedLegalDocumentProcessor/g' app.py
   ```

5. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

6. Access the application at http://localhost:8501

### Running Without Docker Compose

If you prefer to run the Docker container directly:

```bash
docker build -t legal-document-analyzer .
docker run -p 8501:8501 legal-document-analyzer
```

## Manual Setup (without Docker)

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
- Required language packs for Tesseract

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## File Descriptions

- `document_processor.py`: Contains the core functionality for document processing and legal analysis
- `app.py`: Streamlit interface for the application
- `Dockerfile`: Instructions for building the Docker container
- `docker-compose.yml`: Configuration for Docker Compose setup
- `requirements.txt`: Python dependencies

## Troubleshooting

If you encounter memory issues during OCR processing of large documents:
- Increase the memory limit in docker-compose.yml
- Process smaller documents or reduce image DPI settings

If OCR quality is poor:
- Ensure documents are scanned at at least 300 DPI
- Check if the correct language packs are installed for Tesseract

## License

[Your license information here]