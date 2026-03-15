# FEM-to-Neural-Operators Workshop AI Assistant 🧠

A RAG (Retrieval-Augmented Generation) chatbot that contains the full knowledge of the **"From FEM to Neural Operators"** workshop (Bauhaus Spring School 2026). Built with GPT-5.1, e5-large-v2 embeddings, Qdrant vector store, and OpenAI Agents SDK guardrails.

## Features

- 🎓 **Full course knowledge**: Lectures, code tutorials, and video transcripts across FEM, PINNs, and Neural Operators
- 🧠 **GPT-5.1 powered**: Best-in-class reasoning for math, code, and scientific concepts
- 🔍 **Semantic search**: e5-large-v2 embeddings with Qdrant HNSW indexing and MMR retrieval
- 🛡️ **Guardrails**: Input/output validation via OpenAI Agents SDK to keep conversations on-topic
- 🎨 **Premium UI**: Dark-themed Streamlit interface with glassmorphism design
- 📚 **Source attribution**: See which lectures, code files, and transcripts informed each answer
- 🗂️ **Module filtering**: Focus on FEM, PINN, or Neural Operators specifically

## Quick Start

### 1. Create conda environment
```bash
cd RAG_FEM_to_NO
conda create -n rag_workshop python=3.11 -y
conda activate rag_workshop
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Ingest documents (run once)
```bash
python ingest.py
```
This loads all workshop materials, embeds them with e5-large-v2, and stores in Qdrant.

### 5. Launch the chatbot
```bash
streamlit run app.py
```

## Architecture

```
User Query → Input Guardrail (gpt-4o-mini) → Qdrant Retriever (MMR, k=6) 
           → GPT-5.1 RAG Chain → Output Guardrail (gpt-4o-mini) → Response
```

| Component | Technology |
|-----------|------------|
| LLM | GPT-5.1 |
| Embeddings | intfloat/e5-large-v2 (local) |
| Vector Store | Qdrant (local persistent) |
| Guardrails | OpenAI Agents SDK |
| UI | Streamlit |

## Project Structure

```
RAG_FEM_to_NO/
├── app.py              # Streamlit chatbot UI
├── rag_chain.py        # RAG chain (retriever + LLM)
├── guardrails.py       # Input/output guardrails
├── ingest.py           # Document ingestion pipeline
├── requirements.txt    # Dependencies
├── .env.example        # API key template
├── qdrant_data/        # Persistent vector store (created by ingest.py)
└── README.md           # This file
```