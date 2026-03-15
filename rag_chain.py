"""
RAG chain for the FEM-to-Neural-Operators chatbot.

Loads the Qdrant vector store, creates a conversational retrieval chain 
with GPT-5.1, and integrates guardrails.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_data")
COLLECTION_NAME = "fem_to_no_workshop"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
LLM_MODEL = "gpt-5.1"
RETRIEVER_K = 6

SYSTEM_PROMPT = """You are an expert teaching assistant for the "From FEM to Neural Operators" \
workshop (Bauhaus Spring School 2026). You have deep knowledge of:

- **Finite Element Methods (FEM)**: strong and weak forms, Galerkin method, basis functions, \
parent elements, isoparametric mapping, Gauss-Legendre quadrature, assembly of stiffness \
matrices and load vectors, Dirichlet and Neumann boundary conditions, error norms (L2, energy), \
Poisson equation, nonlinear problems, Newton-Raphson method
- **Physics-Informed Neural Networks (PINNs)**: using neural networks to solve PDEs, \
data-driven vs physics-informed loss functions, collocation points, automatic differentiation
- **Neural Operators (DeepONet)**: operator learning, branch network and trunk network \
architecture, learning maps between function spaces, training with MSE loss, \
cantilever beam deflection problems

When answering questions:
1. **Be highly concise and direct** — avoid long preamble or verbose introductions. Give the precise answer immediately.
2. **Be pedagogical but brief** — explain concepts step by step building from fundamentals, but keep explanations compact.
3. **Reference specific sources** — mention relevant lecture names, code files, or transcripts \
when your answer draws from them (e.g., "As covered in FEM Linear 1 - From Strong to Weak Form")
4. **For code questions** — provide clear explanations of the code logic; you can show \
code snippets but explain each part
5. **For math/equations** — ALWAYS use dollar-sign LaTeX delimiters:
   - Inline math: $equation$ (single dollar signs)
   - Display/block math: $$equation$$ (double dollar signs)
   - NEVER use backslash-bracket \\[...\\] or backslash-paren \\(...\\) delimiters
   - Example: "The weak form is $$\\int_a^b u'(x) v'(x) \\, dx = \\int_a^b f(x) v(x) \\, dx$$"
6. **If unsure** — say so honestly rather than guessing
7. **Answer EXACTLY what is asked** — do not volunteer extra information, tangentially \
related facts, or unprompted tutorials.
8. **Stay in scope** — only answer questions related to FEM, PINNs, Neural Operators, \
and related scientific computing topics

Use the following retrieved context to answer the question. If the context doesn't contain \
enough information, say so and provide what you can based on general knowledge of the topics.

Context:
{context}"""


def fix_latex_delimiters(text: str) -> str:
    """Convert LaTeX delimiters to Streamlit-compatible format.
    
    Streamlit renders $...$ (inline) and $$...$$ (block) but NOT
    \\[...\\] or \\(...\\) or \\begin{equation}...\\end{equation}.
    """
    import re
    # Convert \[...\] to $$...$$
    text = re.sub(r'\\\[\s*', '\n$$\n', text)
    text = re.sub(r'\s*\\\]', '\n$$\n', text)
    # Convert \(...\) to $...$
    text = re.sub(r'\\\(\s*', '$', text)
    text = re.sub(r'\s*\\\)', '$', text)
    return text


# ---------------------------------------------------------------------------
# Embedding model (cached singleton)
# ---------------------------------------------------------------------------
_embeddings = None

def get_embeddings():
    """Get or create the embedding model (singleton)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Vector store and retriever
# ---------------------------------------------------------------------------
_qdrant_client = None
_vector_store = None


def get_vector_store():
    """Load the persistent Qdrant vector store (singleton)."""
    global _qdrant_client, _vector_store
    if _vector_store is not None:
        return _vector_store

    qdrant_path = Path(QDRANT_PATH).resolve()
    if not qdrant_path.exists():
        raise FileNotFoundError(
            f"Qdrant data not found at {qdrant_path}. "
            "Please run 'python ingest.py' first."
        )

    _qdrant_client = QdrantClient(path=str(qdrant_path))
    embeddings = get_embeddings()

    _vector_store = QdrantVectorStore(
        client=_qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return _vector_store


def get_retriever(module_filter: str | None = None):
    """
    Create a retriever from the vector store.
    
    Args:
        module_filter: Optional module name to filter results (FEM, NO, PINN)
    """
    vector_store = get_vector_store()

    search_kwargs = {
        "k": RETRIEVER_K,
    }

    # Add module filter if specified
    if module_filter and module_filter != "All Topics":
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.module",
                    match=MatchValue(value=module_filter),
                )
            ]
        )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
    return retriever


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

def format_docs(docs):
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        module = doc.metadata.get("module", "Unknown")
        ctype = doc.metadata.get("content_type", "Unknown")
        formatted.append(
            f"[Source {i}: {source} | Module: {module} | Type: {ctype}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def format_chat_history(chat_history: list[dict]) -> list:
    """Convert chat history dicts to LangChain message objects."""
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def get_rag_chain(module_filter: str | None = None):
    """
    Create the full RAG chain.
    
    Args:
        module_filter: Optional module filter for retrieval
        
    Returns:
        A callable chain that takes {"question": str, "chat_history": list}
        and returns {"answer": str, "source_documents": list}
    """
    retriever = get_retriever(module_filter)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        streaming=True,
    )

    # Build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # The chain
    def invoke(inputs: dict) -> dict:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Format context
        context = format_docs(docs)

        # Format chat history
        history_messages = format_chat_history(chat_history)

        # Run through LLM
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "chat_history": history_messages,
            "question": question,
        })

        return {
            "answer": answer,
            "source_documents": docs,
        }

    def stream(inputs: dict):
        """Stream the response token by token, also return source docs."""
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Format context
        context = format_docs(docs)

        # Format chat history
        history_messages = format_chat_history(chat_history)

        # Stream through LLM
        chain = prompt | llm

        for chunk in chain.stream({
            "context": context,
            "chat_history": history_messages,
            "question": question,
        }):
            if hasattr(chunk, "content"):
                yield chunk.content

        # Yield source documents at the end (as a special marker)
        yield {"__source_documents__": docs}

    return type("RAGChain", (), {"invoke": staticmethod(invoke), "stream": staticmethod(stream)})()
