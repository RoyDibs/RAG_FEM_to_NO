"""
Document ingestion pipeline for the FEM-to-Neural-Operators RAG chatbot.

Loads PDFs, MATLAB .m files, Jupyter notebooks, and DOCX files from the
Data_for_RAG directory, chunks them, embeds with e5-large-v2, and stores
in a persistent Qdrant vector store.

Usage:
    python ingest.py
"""

import os
import sys
import json
import glob
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import nbformat

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "RAG_workshop", "Data_for_RAG")
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_data")
COLLECTION_NAME = "fem_to_no_workshop"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# File types to skip (binary / non-text)
SKIP_EXTENSIONS = {".mat", ".png", ".mp4", ".DS_Store"}

# Module directories
MODULES = ["FEM", "NO", "PINN"]

# Content type mapping based on subdirectory names
CONTENT_TYPE_MAP = {
    "code": "code",
    "lectures": "lecture",
    "lecture": "lecture",
    "transcripts from videolectures": "transcript",
    "transcripts for video lectures": "transcript",
    "transcript": "transcript",
}


# ---------------------------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------------------------

def load_pdf(filepath: str, metadata: dict) -> list[Document]:
    """Load a PDF file and return documents with metadata."""
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        print(f"  ⚠️  Error loading PDF {filepath}: {e}")
        return []


def load_matlab_file(filepath: str, metadata: dict) -> list[Document]:
    """Load a MATLAB .m file as plain text, preserving comments."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return []

        # Add file header for context
        filename = os.path.basename(filepath)
        header = f"% MATLAB source file: {filename}\n% Module: {metadata.get('module', 'unknown')}\n\n"
        content = header + content

        doc = Document(page_content=content, metadata=metadata)
        return [doc]
    except Exception as e:
        print(f"  ⚠️  Error loading MATLAB file {filepath}: {e}")
        return []


def load_notebook(filepath: str, metadata: dict) -> list[Document]:
    """Load a Jupyter notebook, extracting markdown and code cells."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        docs = []
        filename = os.path.basename(filepath)

        for i, cell in enumerate(nb.cells):
            if cell.cell_type in ("markdown", "code"):
                content = cell.source.strip()
                if not content:
                    continue

                # Add context header
                cell_type_label = "Markdown" if cell.cell_type == "markdown" else "Python Code"
                header = f"# Jupyter Notebook: {filename} | Cell {i+1} ({cell_type_label})\n\n"

                cell_metadata = metadata.copy()
                cell_metadata["cell_type"] = cell.cell_type
                cell_metadata["cell_index"] = i

                doc = Document(
                    page_content=header + content,
                    metadata=cell_metadata,
                )
                docs.append(doc)

        return docs
    except Exception as e:
        print(f"  ⚠️  Error loading notebook {filepath}: {e}")
        return []


def load_docx(filepath: str, metadata: dict) -> list[Document]:
    """Load a DOCX file."""
    try:
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        print(f"  ⚠️  Error loading DOCX {filepath}: {e}")
        return []


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------

def detect_content_type(subdir: str) -> str:
    """Detect content type from subdirectory name."""
    normalized = subdir.lower().strip()
    return CONTENT_TYPE_MAP.get(normalized, "unknown")


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def load_all_documents() -> list[Document]:
    """Walk the Data_for_RAG directory and load all text-bearing files."""
    all_docs = []
    data_path = Path(DATA_DIR).resolve()

    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        sys.exit(1)

    print(f"📂 Data directory: {data_path}")
    print()

    for module in MODULES:
        module_path = data_path / module
        if not module_path.exists():
            print(f"  ⚠️  Module directory not found: {module_path}")
            continue

        print(f"📦 Module: {module}")

        for subdir in sorted(module_path.iterdir()):
            if not subdir.is_dir():
                continue

            content_type = detect_content_type(subdir.name)
            print(f"  📁 {subdir.name} (type: {content_type})")

            for filepath in sorted(subdir.iterdir()):
                if not filepath.is_file():
                    continue

                ext = filepath.suffix.lower()

                if ext in SKIP_EXTENSIONS or filepath.name.startswith("."):
                    print(f"    ⏭️  Skipping: {filepath.name}")
                    continue

                metadata = {
                    "module": module,
                    "content_type": content_type,
                    "source_file": filepath.name,
                    "file_type": ext,
                    "source_path": str(filepath),
                }

                docs = []
                if ext == ".pdf":
                    docs = load_pdf(str(filepath), metadata)
                elif ext == ".m":
                    docs = load_matlab_file(str(filepath), metadata)
                elif ext == ".ipynb":
                    docs = load_notebook(str(filepath), metadata)
                elif ext == ".docx":
                    docs = load_docx(str(filepath), metadata)
                else:
                    print(f"    ⏭️  Unknown format: {filepath.name}")
                    continue

                print(f"    ✅ {filepath.name} → {len(docs)} document(s)")
                all_docs.extend(docs)

        print()

    return all_docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks with appropriate splitters."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,  # Works reasonably for MATLAB too
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunked = []
    for doc in docs:
        if doc.metadata.get("content_type") == "code":
            chunks = code_splitter.split_documents([doc])
        else:
            chunks = text_splitter.split_documents([doc])
        chunked.extend(chunks)

    return chunked


def create_embeddings():
    """Create the HuggingFace e5-large-v2 embedding model."""
    print("🧠 Loading e5-large-v2 embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )
    print("  ✅ Embedding model loaded")
    return embeddings


def create_vector_store(chunks: list[Document], embeddings) -> None:
    """Create and persist a Qdrant vector store."""
    qdrant_path = Path(QDRANT_PATH).resolve()
    print(f"🗄️  Creating Qdrant vector store at: {qdrant_path}")

    # Get embedding dimension
    sample_embedding = embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)
    print(f"  📐 Embedding dimension: {embedding_dim}")

    # Initialize Qdrant client with local persistent storage
    client = QdrantClient(path=str(qdrant_path))

    # Delete existing collection if it exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"  🗑️  Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"  ✅ Collection created: {COLLECTION_NAME}")

    # Add documents in batches
    print(f"  📥 Embedding and storing {len(chunks)} chunks...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Add in batches to show progress
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        progress = min(i + batch_size, len(chunks))
        print(f"    📊 {progress}/{len(chunks)} chunks stored")

    print(f"  ✅ Vector store created with {len(chunks)} chunks")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("🚀 FEM-to-Neural-Operators RAG Ingestion Pipeline")
    print("=" * 60)
    print()

    # 1. Load documents
    print("Step 1: Loading documents...")
    docs = load_all_documents()
    print(f"📄 Total documents loaded: {len(docs)}")
    print()

    if not docs:
        print("❌ No documents found. Check the data directory path.")
        sys.exit(1)

    # 2. Chunk documents
    print("Step 2: Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"🧩 Total chunks created: {len(chunks)}")
    print()

    # Print summary by module and content type
    summary = {}
    for chunk in chunks:
        module = chunk.metadata.get("module", "unknown")
        ctype = chunk.metadata.get("content_type", "unknown")
        key = f"{module}/{ctype}"
        summary[key] = summary.get(key, 0) + 1

    print("📊 Chunk distribution:")
    for key in sorted(summary.keys()):
        print(f"  {key}: {summary[key]} chunks")
    print()

    # 3. Create embeddings and vector store
    print("Step 3: Creating embeddings and vector store...")
    embeddings = create_embeddings()
    create_vector_store(chunks, embeddings)

    print()
    print("=" * 60)
    print("✅ Ingestion complete!")
    print(f"   Qdrant data stored at: {Path(QDRANT_PATH).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
