# 📚 Learn RAG: Cloud-Native RAG Workshop Application

A production-grade, beginner-friendly **Retrieval-Augmented Generation (RAG)** demonstration web application built with Python, Streamlit, LangChain, Qdrant Cloud, and OpenRouter.

Upload any text-based PDF, index its contents into a managed cloud vector database, and ask natural language questions. Answers are generated using OpenRouter (`recursive/flint-v1.0:free`), grounded **exclusively** in the uploaded document, and accompanied by **exact page-number citations** and an interactive chunk inspection panel.

---

## 🚀 Key Features

- **☁️ 100% Cloud-Native Vector Store:** Connects directly to **Qdrant Cloud** free tier. No local Docker installation required.
- **📄 Deterministic Document Ingestion:** Uses SHA-256 binary hashing to assign a unique `doc_id` to each uploaded PDF, preventing redundant indexing and document mixing.
- **🔢 1-Indexed Human Page Citations:** Converts 0-indexed PDF metadata into standard 1-indexed page numbers displayed as verified citation badges (`document.pdf (Pages 2, 5)`).
- **✂️ Optimized Semantic Chunking:** Chunks document pages with `RecursiveCharacterTextSplitter` (1,000 characters chunk size, 400 characters overlap) to preserve context continuity across boundaries.
- **🔒 Anti-Hallucination Guardrails:** Strict prompt engineering ensures the LLM relies solely on retrieved excerpts. If the information is not present, it explicitly responds:
  > *"The provided document does not contain information to answer this question."*
- **🩺 Retrieval Diagnostics:** An expandable inspector showing exact retrieved text chunks, source pages, and cosine similarity scores.
- **🧪 Automated Test Suite:** Pre-configured unit tests in `pytest` to test chunking, citation deduplication, and grounding refusal without consuming API credits.

---

## 🏛️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RAG WORKFLOW                                │
│                                                                             │
│ [PHASE A: DOCUMENT INGESTION & CLOUD INDEXING]                             │
│  1. Upload PDF ──► 2. PyPDFLoader ──► 3. RecursiveTextSplitter              │
│    (Streamlit)      (Extract Pages     (1,000 chars size,                   │
│                      + 1-idx Page)      400 chars overlap)                  │
│                                              │                              │
│                                              ▼                              │
│                                       4. Embeddings ──► 5. Qdrant Cloud     │
│                                       (all-MiniLM-L6      (learning_rag     │
│                                        384 dimensions)     Collection)      │
│                                                                             │
│ [PHASE B: RETRIEVAL & GROUNDED ANSWER GENERATION]                           │
│  6. User Query ──► 7. Vectorize Query ──► 8. Cosine Similarity              │
│    ("What is...")   (all-MiniLM-L6)          (Scoped to doc_id)             │
│                                                      │                      │
│                                                      ▼                      │
│                                              9. OpenRouter LLM              │
│                                            (recursive/flint-v1.0)           │
│                                                      │                      │
│                                                      ▼                      │
│                                            10. Grounded Answer +            │
│                                                Page Citations & Chunks      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
learn-rag/
├── app.py                      # Streamlit UI & interactive chat workflow
├── rag/                        # Core modular RAG package
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Environment configuration and validation
│   ├── ingestion.py            # PDF loading, SHA-256 hashing, chunking, Qdrant upload
│   ├── retrieval.py            # Embedding cache, scoped similarity search, citations
│   └── generation.py           # Grounded prompt engineering and OpenRouter LLM call
├── tests/                      # Automated test suite
│   ├── __init__.py             # Test package initializer
│   ├── test_rag_pipeline.py    # Unit tests for chunking, citations, and grounding
│   └── validate_cloud_setup.py # End-to-end live cloud infrastructure test
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment configuration
├── .env                        # Local configuration with cloud credentials (gitignored)
├── .gitignore                  # Git ignore rules for secrets and virtualenvs
├── README.md                   # Project overview and workshop guide
└── implementation.md           # Step-by-step implementation guide
```

---

## 🛠️ Prerequisites & Setup

### 1. Free Account Setup
1. **Qdrant Cloud:** Sign up at [https://cloud.qdrant.io/](https://cloud.qdrant.io/), create a free 1GB cluster, and generate an API key.
2. **OpenRouter:** Sign up at [https://openrouter.ai/](https://openrouter.ai/) and generate a secret key at [https://openrouter.ai/keys](https://openrouter.ai/keys).

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:

```env
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=learning_rag
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_api_key_here
OPENROUTER_MODEL=recursive/flint-v1.0:free
```

### 3. Installation (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Testing

### Automated Unit Tests (Offline / No API costs)
```powershell
pytest tests/ -v
```

### End-to-End Cloud Validation
Verify your Qdrant Cloud cluster and OpenRouter connection:
```powershell
python tests/validate_cloud_setup.py
```

---

## 🖥️ Running the Application

```powershell
streamlit run app.py
```

The application will start and open in your default browser at `http://localhost:8501`.

---

## 🎓 Workshop Verification Scenarios

1. **Verify Indexing & Citations:** Upload a document (e.g. `nodejs.pdf`), click **Index to Cloud**, and ask a question from page 8. Confirm page citation badge appears (`nodejs.pdf (Page 8)`).
2. **Inspect Retrieved Chunks:** Expand the **Retrieval Diagnostics** toggle to verify 4 passages were retrieved with cosine similarity scores.
3. **Anti-Hallucination Test:** Ask an unrelated question (e.g. *"What is the boiling point of tungsten?"*). Verify the system refuses to extrapolate and replies:
   > *"The provided document does not contain information to answer this question."*
