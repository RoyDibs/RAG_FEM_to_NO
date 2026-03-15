"""
FEM-to-Neural-Operators Workshop AI Assistant

A premium Streamlit chatbot powered by GPT-5.1, e5-large-v2 embeddings, 
Qdrant vector store, and OpenAI Agents SDK guardrails.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Support Streamlit Cloud secrets (preferred) or local .env file
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Workshop AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #111827 50%, #0f172a 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 0;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.08);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    /* User messages */
    [data-testid="stChatMessage"][data-testid*="user"] {
        border-color: rgba(99, 102, 241, 0.15);
        background: rgba(99, 102, 241, 0.05);
    }

    /* Chat input styling */
    [data-testid="stChatInput"] {
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
    }

    [data-testid="stChatInput"] textarea {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Expander styling for sources */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 8px;
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(99, 102, 241, 0.1));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 8px;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(99, 102, 241, 0.2));
        border-color: rgba(56, 189, 248, 0.4);
    }

    /* Selectbox styling */
    .stSelectbox label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Warning/info boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* Badge styling */
    .powered-badge {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(99, 102, 241, 0.08));
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.75rem;
        color: #64748b;
        text-align: center;
        margin-top: 2rem;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.06), rgba(99, 102, 241, 0.06));
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .welcome-card h3 {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Source item styling */
    .source-item {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.08);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
    }

    .source-module {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    .source-type {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* Divider */
    .sidebar-divider {
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🧠 Workshop AI Assistant")
    st.markdown("*From FEM to Neural Operators*")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This AI assistant has knowledge of the complete 
        **From FEM to Neural Operators** workshop content:

        - 📐 **FEM**: 5 lectures, 15 MATLAB code files, 9 transcripts
        - 🧪 **PINN**: 1 lecture, 2 Jupyter notebooks, 2 transcripts  
        - 🤖 **Neural Operators**: 1 lecture, 1 DeepONet notebook, 2 transcripts

        Ask about theory, code, math, or concepts!
        """)



# ---------------------------------------------------------------------------
# Initialize RAG chain (lazy loading)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_rag_chain(module_filter: str):
    """Load the RAG chain (cached per module filter)."""
    from rag_chain import get_rag_chain
    return get_rag_chain(module_filter)


def fix_latex(text: str) -> str:
    """Fix LaTeX delimiters for Streamlit rendering."""
    from rag_chain import fix_latex_delimiters
    return fix_latex_delimiters(text)


# ---------------------------------------------------------------------------
# Pre-load RAG chain at startup (so spinner doesn't appear in chat)
# ---------------------------------------------------------------------------
with st.spinner("🔄 Loading AI assistant..."):
    _chain = load_rag_chain("All Topics")


def get_chain():
    """Get the cached RAG chain."""
    return _chain


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Welcome message (only shown when no messages)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome!</h3>
        <p>I'm your AI assistant for the <strong>From FEM to Neural Operators</strong> 
        workshop. I can help you with:</p>
        <ul>
            <li>📐 <strong>FEM concepts</strong> — weak forms, assembly, boundary conditions, error norms</li>
            <li>🧪 <strong>PINNs</strong> — physics-informed neural networks, loss functions, training</li>
            <li>🤖 <strong>Neural Operators</strong> — DeepONet architecture, operator learning</li>
            <li>💻 <strong>Code explanations</strong> — MATLAB FEM code, Python/JAX notebooks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🧠"):
        st.markdown(fix_latex(message["content"]))

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Sources used"):
                for src in message["sources"]:
                    module_tag = src.get("module", "?")
                    ctype_tag = src.get("content_type", "?")
                    filename = src.get("source_file", "Unknown")
                    # Clean up filename for UI: remove common extensions
                    clean_name = filename.replace(".pdf", "").replace(".m", "").replace(".ipynb", "").replace(".docx", "")
                    st.markdown(
                        f'<div class="source-item">'
                        f'<span class="source-module">{module_tag}</span>'
                        f'<span class="source-type">{ctype_tag}</span> '
                        f'{clean_name}</div>',
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Ask about FEM, PINNs, or Neural Operators..."):
    # Display user message
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run input guardrail
    guardrail_passed = True
    guardrail_msg = ""

    try:
        from guardrails import check_input_guardrail
        guardrail_passed, guardrail_msg = check_input_guardrail(user_input)
    except Exception as e:
        # If guardrails module fails to import or run, allow through
        guardrail_passed = True
        guardrail_msg = f"Guardrail check skipped: {e}"

    if not guardrail_passed:
        # Show blocked message
        with st.chat_message("assistant", avatar="🧠"):
            blocked_msg = (
                "⚠️ I'm specifically designed to help with the **From FEM to Neural Operators** "
                "workshop content. Your question appears to be outside my area of expertise.\n\n"
                "**I can help with:**\n"
                "- Finite Element Methods (FEM)\n"
                "- Physics-Informed Neural Networks (PINNs)\n"
                "- Neural Operators (DeepONet)\n"
                "- Related scientific computing topics\n\n"
                "Please try rephrasing your question!"
            )
            st.markdown(blocked_msg)

        st.session_state.messages.append({
            "role": "assistant",
            "content": blocked_msg,
            "sources": [],
        })
    else:
        # Get RAG response
        with st.chat_message("assistant", avatar="🧠"):
            chain = get_chain()

            try:
                # Use invoke for simplicity & source document access
                with st.spinner("🤔 Thinking..."):
                    result = chain.invoke({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history,
                    })

                answer = fix_latex(result["answer"])
                source_docs = result.get("source_documents", [])

                # Run output guardrail
                try:
                    from guardrails import check_output_guardrail
                    output_ok, output_msg = check_output_guardrail(answer)
                    if not output_ok:
                        answer += (
                            "\n\n---\n*⚠️ Note: This response was flagged by the quality "
                            "check system. Please verify the information independently.*"
                        )
                except Exception:
                    pass  # Fail-open on output guardrail errors

                # Display answer
                st.markdown(answer)

                # Show sources
                sources_meta = []
                if source_docs:
                    # Deduplicate sources by filename
                    seen = set()
                    unique_sources = []
                    for doc in source_docs:
                        fname = doc.metadata.get("source_file", "Unknown")
                        if fname not in seen:
                            seen.add(fname)
                            unique_sources.append(doc.metadata)

                    with st.expander("📚 Sources used"):
                        for src in unique_sources:
                            module_tag = src.get("module", "?")
                            ctype_tag = src.get("content_type", "?")
                            filename = src.get("source_file", "Unknown")
                            # Clean up filename for UI
                            clean_name = filename.replace(".pdf", "").replace(".m", "").replace(".ipynb", "").replace(".docx", "")
                            st.markdown(
                                f'<div class="source-item">'
                                f'<span class="source-module">{module_tag}</span>'
                                f'<span class="source-type">{ctype_tag}</span> '
                                f'{clean_name}</div>',
                                unsafe_allow_html=True,
                            )
                    sources_meta = unique_sources

                # Store in session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_meta,
                })

                # Update chat history for context
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                })

                # Keep chat history manageable (last 10 turns)
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

            except FileNotFoundError as e:
                st.error(
                    "📁 **Vector store not found.** Please run `python ingest.py` first "
                    "to build the knowledge base."
                )
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
