# HR Chatbot - Production Setup Guide

## Overview
Production-ready HR chatbot system with RAG (Retrieval-Augmented Generation) using BGE-large embeddings, PostgreSQL with pgvector, and OpenAI for LLM inference.

## System Architecture
- **Frontend**: Streamlit web application
- **Backend**: Python FastAPI-like architecture
- **Vector Database**: PostgreSQL with pgvector extension
- **Embeddings**: BGE-large-en-v1.5 (1024 dimensions)
- **LLM**: OpenAI (configurable model)

## Prerequisites

### 1. Database Setup
```bash
# Install PostgreSQL with pgvector
# For Windows: Download from https://www.postgresql.org/download/windows/
# For Linux: sudo apt-get install postgresql postgresql-contrib

# Install pgvector extension
# Follow: https://github.com/pgvector/pgvector#installation
```


On Windows: Download and install from respective websites, ensure they are in PATH.

### 2. OpenAI API Key
Obtain an API key from https://platform.openai.com/account/api-keys

### 3. Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Database Configuration
Edit `config.env`:
```env
# PostgreSQL Configuration
PGHOST=localhost
PGPORT=5433
PGDATABASE=test
PGUSER=postgres
PGPASSWORD=password

# OpenAI Configuration
API_KEY=your_openai_api_key_here
MODEL=gpt-4o

# RAG Configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
SIMILARITY_THRESHOLD=0.6
RETRIEVAL_K=5

# PDF Knowledge Base
PDF_PATH=../knowledge_base
```

Note: CHUNK_SIZE and CHUNK_OVERLAP are not used in the current semantic chunking implementation.

### 2. Knowledge Base Setup
Place your PDF documents in the directory specified by PDF_PATH in config.env (default: `../knowledge_base/`).

The system uses the unstructured library for intelligent PDF parsing and semantic chunking.

## Running the Application

### Production Deployment

1. **Start PostgreSQL** (ensure pgvector extension is available)

2. **Initialize and Run the Application**:
```bash
cd /path/to/HR_CHATBOT/psql_chatbot
streamlit run new_app.py --server.port 8501 --server.address 0.0.0.0
```

3. **Access the Application**:
Open a web browser and navigate to http://localhost:8501 (or the IP address of the server if running remotely).

### Development Mode
```bash
# Run with debug logging
STREAMLIT_LOGGER_LEVEL=debug streamlit run new_app.py
```

## System Features

### Automatic Setup
- ✅ **Auto-detects BGE-large embedding dimensions** (1024)
- ✅ **Automatically initializes database tables**
- ✅ **Validates system initialization** before serving requests
- ✅ **Handles PDF loading and semantic chunking** using unstructured library

### Production Ready
- ✅ **Error handling and logging**
- ✅ **Session management with database persistence**
- ✅ **Vector similarity search**
- ✅ **Chat history persistence**
- ✅ **Responsive UI with home page and chat interface**
- ✅ **FAQ suggestions**
- ✅ **Streaming responses**

## Troubleshooting

### Database Issues
- **Connection Error**: Check PostgreSQL is running and credentials in `config.env`
- **pgvector Extension**: Ensure `CREATE EXTENSION vector;` works in your database

### Model Issues
- **OpenAI Connection**: Ensure API_KEY is valid and you have credits in your OpenAI account
- **BGE Model Download**: First run will download the BGE-large model (~1.2GB)

### PDF Processing Issues
- **Missing Dependencies**: Ensure Tesseract and Poppler are installed
- **OCR Errors**: Check Tesseract configuration for scanned PDFs

### Performance Optimization
- **Vector Index**: Automatically created for fast similarity search
- **Batch Processing**: Embeddings are processed in batches
- **Connection Pooling**: Database connections are managed efficiently

## File Structure
```
psql_chatbot/
├── new_app.py          # Main Streamlit application
├── rag_system.py       # RAG system with vector search
├── llm_client.py       # LLM client with chat management
├── config.env          # Configuration file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Production Checklist
- [ ] PostgreSQL with pgvector is running
- [ ] OpenAI API key is configured in `config.env`
- [ ] `config.env` is properly configured
- [ ] Knowledge base PDFs are in place
- [ ] Python environment and dependencies installed
- [ ] System initializes successfully (check logs)

## Support
The system includes comprehensive logging and error handling. Check the Streamlit interface for status messages and initialization feedback. 