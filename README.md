# 🎥 YouTube Notes RAG System

A Retrieval-Augmented Generation (RAG) system that lets you:
- Fetch YouTube video transcripts
- Generate embeddings and store them locally
- Query videos using natural language
- Get AI-powered summaries and Q&A

## Features
- **Transcript Extraction**: Automatically fetch YouTube video transcripts
- **Local Vector Database**: ChromaDB for efficient similarity search
- **LLM Integration**: Flan-T5 for summarization and Q&A
- **Strict Context-Only Answers**: Prevents hallucinations
- **Modular Architecture**: Easily swap components (database, LLM, etc.)
- **Manual Transcript Paste**: Fallback when YouTube blocks IP

## Tech Stack
- **Backend**: Python 3.11+
- **Vector DB**: ChromaDB
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM**: Google Flan-T5-Base
- **UI**: Streamlit

## Installation

1. **Prerequisites**:
   - Python 3.11+

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
youtube-notes/
├── database/
│   ├── databaseInterface.py
│   └── chroma.py
├── utils/
│   └── youtube_utils.py
├── youtube_notes.py
├── app.py
└── requirements.txt
```

## Usage

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```

2. **Process a video**:
   - Enter YouTube Video ID (e.g., `pNJUyol15Jw`)
   - Click "Process Video" to generate embeddings and summary
   - Or use "Paste Transcript" mode as a fallback

3. **Ask questions**:
   - Type natural language questions about the video
   - Get answers strictly based on the transcript