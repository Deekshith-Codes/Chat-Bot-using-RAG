"""
Learn RAG — Modern AI Document Assistant
=========================================
Production-grade, grounded question-answering application backed by
Qdrant Cloud vector search, Hugging Face embeddings, and OpenRouter LLMs.
"""

import os
from pathlib import Path
import streamlit as st

from rag.config import get_config, validate_config
from rag.ingestion import (
    compute_document_id,
    load_pdf_from_bytes,
    split_document_pages,
    get_qdrant_client,
    ensure_collection_exists,
    is_document_indexed,
    delete_document_vectors,
    index_chunks_to_qdrant,
)
from rag.retrieval import (
    get_embedding_model,
    retrieve_relevant_chunks,
    format_citations,
)
from rag.generation import generate_grounded_answer


# -----------------------------------------------------------------------------
# Streamlit Application Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Learn RAG — AI Document Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Load Modern AI SaaS CSS Design System
# -----------------------------------------------------------------------------
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "show_diagnostics" not in st.session_state:
    st.session_state.show_diagnostics = False

# -----------------------------------------------------------------------------
# Environment & Configuration Verification
# -----------------------------------------------------------------------------
config = get_config()
is_config_valid, config_issues = validate_config(config)

# -----------------------------------------------------------------------------
# Left Sidebar: Navigation, Document Management & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Brand Header
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon-box">⚡</div>
            <div class="brand-text-box">
                <h2>Learn RAG <span class="brand-badge">SaaS</span></h2>
                <p>Cloud Document Intelligence</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Primary Action: New Chat
    if st.button("➕ New Chat Session", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown('<div class="sidebar-section-title">📂 Document Context</div>', unsafe_allow_html=True)

    # 3. Active Document Card & Controls
    if st.session_state.active_document:
        doc = st.session_state.active_document
        doc_name = doc.get("filename", "document.pdf")
        doc_id = doc.get("doc_id", "N/A")
        total_pages = doc.get("total_pages", "N/A")
        total_chunks = doc.get("total_chunks", "N/A")

        st.markdown(
            f"""
            <div class="modern-card">
                <div class="doc-card-header">
                    <span class="doc-name" title="{doc_name}">📄 {doc_name}</span>
                    <span class="status-pill status-pill-green">
                        <span class="status-dot status-dot-green"></span> Ready
                    </span>
                </div>
                <div class="doc-meta">
                    <span>📑 {total_pages} pages</span>
                    <span>🧩 {total_chunks} chunks</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_switch, col_del = st.columns(2)
        with col_switch:
            if st.button("🔄 Replace", use_container_width=True, help="Switch active PDF"):
                st.session_state.active_document = None
                st.rerun()

        with col_del:
            if st.button("🗑️ Delete", use_container_width=True, help="Remove vectors from Qdrant Cloud"):
                with st.spinner("Deleting vectors..."):
                    try:
                        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                        del_count = delete_document_vectors(
                            client, config.qdrant_collection_name, doc["doc_id"]
                        )
                        st.session_state.active_document = None
                        st.session_state.messages = []
                        st.success(f"Removed {del_count} vectors from cloud!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Deletion failed: {err}")
    else:
        st.markdown(
            """
            <div class="modern-card" style="text-align: center; padding: 1.1rem 0.8rem;">
                <p style="margin: 0; font-size: 0.84rem; color: #94A3B8;">
                    No PDF currently active.<br>
                    <span style="font-size: 0.76rem; color: #64748B;">Upload a document in the main panel to begin.</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. System & Cloud Infrastructure Status
    st.markdown('<div class="sidebar-section-title">☁️ Infrastructure</div>', unsafe_allow_html=True)
    
    # Qdrant Cloud pill
    if config.qdrant_url and "localhost" not in config.qdrant_url and config.qdrant_api_key:
        cluster_host = config.qdrant_url.split("//")[-1].split(":")[0]
        st.markdown(
            f"""
            <div class="modern-card" style="padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.8rem; font-weight: 500; color: #F8FAFC;">Qdrant Cloud</span>
                    <span class="status-pill status-pill-green">
                        <span class="status-dot status-dot-green"></span> Connected
                    </span>
                </div>
                <div style="font-size: 0.72rem; color: #64748B; margin-top: 3px; font-family: monospace;">
                    {cluster_host[:26]}...
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="modern-card" style="padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                <span class="status-pill" style="background: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.25);">
                    ⚠️ Qdrant Cloud Offline
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # OpenRouter LLM pill
    if config.openrouter_api_key:
        model_display = config.openrouter_model.split("/")[-1]
        st.markdown(
            f"""
            <div class="modern-card" style="padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.8rem; font-weight: 500; color: #F8FAFC;">OpenRouter LLM</span>
                    <span class="status-pill status-pill-purple">
                        <span class="status-dot status-dot-purple"></span> Active
                    </span>
                </div>
                <div style="font-size: 0.72rem; color: #818CF8; margin-top: 3px; font-family: monospace;">
                    {model_display}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="modern-card" style="padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                <span class="status-pill" style="background: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.25);">
                    ⚠️ OpenRouter API Key Missing
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("⚙️ System Specs", expanded=False):
        st.markdown(
            f"""
            - **Collection:** `{config.qdrant_collection_name}`
            - **Embeddings:** `{config.embedding_model_name}` (384d)
            - **Chunk Size:** `{config.chunk_size}` chars
            - **Chunk Overlap:** `{config.chunk_overlap}` chars
            - **Top-K Retrieval:** `{config.top_k}`
            """
        )

    if not is_config_valid:
        with st.expander("⚠️ Configuration Checklist", expanded=True):
            for issue in config_issues:
                st.caption(f"• {issue}")
            st.caption("Update your `.env` file to resolve these issues.")

    # 5. Developer & Session Settings
    st.markdown('<div class="sidebar-section-title">🛠️ Settings</div>', unsafe_allow_html=True)
    st.session_state.show_diagnostics = st.toggle(
        "Retrieval Diagnostics",
        value=st.session_state.show_diagnostics,
        help="Inspect retrieved text passages, cosine similarity scores, and source pages.",
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Workspace: Top Navigation & Status Bar
# -----------------------------------------------------------------------------
active_name = (
    st.session_state.active_document["filename"]
    if st.session_state.active_document
    else "No Document Loaded"
)

st.markdown(
    f"""
    <div class="workspace-top-bar">
        <div class="top-bar-left">
            <span class="active-doc-badge">
                <span>📄</span>
                <strong>{active_name}</strong>
            </span>
        </div>
        <div class="top-bar-right">
            <span class="status-pill status-pill-green">
                <span class="status-dot status-dot-green"></span> Qdrant Cloud
            </span>
            <span class="status-pill status-pill-blue">
                ⚡ {config.openrouter_model.split('/')[-1]}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# If configuration is invalid, halt early with clear checklist
if not is_config_valid:
    st.warning("⚠️ **Application Setup Required:** Please verify your credentials in `.env` before proceeding.")
    with st.expander("📋 View Missing Configuration Details", expanded=True):
        for issue in config_issues:
            st.error(issue)
    st.stop()


# -----------------------------------------------------------------------------
# Section 1: PDF Document Upload & Ingestion Accordion
# -----------------------------------------------------------------------------
upload_expanded = st.session_state.active_document is None

with st.expander("📤 Document Ingestion & Vector Indexing", expanded=upload_expanded):
    uploaded_file = st.file_uploader(
        "Upload a PDF document to index into Qdrant Cloud:",
        type=["pdf"],
        help="Select any text-based PDF (documentation, technical report, lecture notes, textbook).",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        doc_id = compute_document_id(file_bytes, filename)
        file_size_kb = len(file_bytes) / 1024

        try:
            qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
            already_indexed, existing_chunks = is_document_indexed(
                qdrant_client, config.qdrant_collection_name, doc_id
            )
        except Exception as conn_err:
            st.error(f"❌ Failed to reach Qdrant Cloud: {conn_err}")
            already_indexed, existing_chunks = False, 0

        col_status, col_action = st.columns([3, 1.2])

        with col_status:
            if already_indexed:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 0.75rem 1rem;">
                        <span style="color: #34D399; font-weight: 600;">✓ Document Already Indexed in Qdrant Cloud</span><br>
                        <span style="font-size: 0.82rem; color: #CBD5E1;">
                            <code>{filename}</code> ({file_size_kb:.1f} KB) &bull; {existing_chunks} vector chunks stored &bull; ID: <code>{doc_id}</code>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 0.75rem 1rem;">
                        <span style="color: #A5B4FC; font-weight: 600;">📄 Ready to Process & Index</span><br>
                        <span style="font-size: 0.82rem; color: #CBD5E1;">
                            <code>{filename}</code> ({file_size_kb:.1f} KB) &bull; Computed Doc ID: <code>{doc_id}</code>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_action:
            if already_indexed:
                if not st.session_state.active_document or st.session_state.active_document.get("doc_id") != doc_id:
                    if st.button("👉 Start Chatting", use_container_width=True, type="primary"):
                        st.session_state.active_document = {
                            "doc_id": doc_id,
                            "filename": filename,
                            "total_chunks": existing_chunks,
                        }
                        st.rerun()
                
                reindex_pressed = st.button("🔄 Re-Index", use_container_width=True)
                start_indexing = reindex_pressed
            else:
                start_indexing = st.button("🚀 Index to Cloud", use_container_width=True, type="primary")

        if start_indexing:
            with st.status("Executing Cloud RAG Ingestion Pipeline...", expanded=True) as status:
                try:
                    status.write("📄 **1/4 Extracting Text:** Parsing PDF and preserving 1-indexed page metadata...")
                    pages = load_pdf_from_bytes(file_bytes, filename)
                    status.write(f"✓ Extracted {len(pages)} page(s) successfully.")

                    status.write(
                        f"✂️ **2/4 Chunking:** Splitting into segments (Chunk size: {config.chunk_size}, overlap: {config.chunk_overlap})..."
                    )
                    chunks = split_document_pages(
                        documents=pages,
                        doc_id=doc_id,
                        filename=filename,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                    )
                    status.write(f"✓ Generated {len(chunks)} chunks with document scoping.")

                    status.write(f"🧠 **3/4 Embeddings:** Initializing `{config.embedding_model_name}`...")
                    embeddings = get_embedding_model(config.embedding_model_name)
                    status.write("✓ Embeddings model ready (384-dimensional dense vectors).")

                    status.write(
                        f"☁️ **4/4 Qdrant Cloud:** Storing vectors in collection `{config.qdrant_collection_name}`..."
                    )
                    indexed_count = index_chunks_to_qdrant(
                        client=qdrant_client,
                        embeddings_model=embeddings,
                        collection_name=config.qdrant_collection_name,
                        chunks=chunks,
                        doc_id=doc_id,
                        overwrite_existing=True,
                    )

                    st.session_state.active_document = {
                        "doc_id": doc_id,
                        "filename": filename,
                        "total_pages": len(pages),
                        "total_chunks": indexed_count,
                    }

                    status.update(
                        label=f"🎉 Successfully indexed '{filename}' ({indexed_count} chunks) in Qdrant Cloud!",
                        state="complete",
                        expanded=False,
                    )
                    st.toast(f"'{filename}' indexed successfully!", icon="✅")
                    st.rerun()

                except Exception as exc:
                    status.update(label=f"❌ Ingestion failed: {exc}", state="error")
                    st.error(f"Error during ingestion: {exc}")


# -----------------------------------------------------------------------------
# Section 2: Conversational RAG Workspace
# -----------------------------------------------------------------------------

# Handle pending query from suggestion chips or direct input
query_to_execute = None
if st.session_state.pending_query:
    query_to_execute = st.session_state.pending_query
    st.session_state.pending_query = None

# A. Empty State Hero (when no messages exist)
if not st.session_state.messages:
    if st.session_state.active_document:
        doc = st.session_state.active_document
        st.markdown(
            f"""
            <div class="hero-container">
                <div class="hero-icon-wrapper">📚</div>
                <div class="hero-title">Ask anything about {doc['filename']}</div>
                <div class="hero-subtitle">
                    Answers are strictly grounded in your document's vectors stored in Qdrant Cloud,
                    with exact source page citations and zero hallucinated facts.
                </div>
                <div class="suggestion-chips-title">Suggested Inquiries:</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 Summarize the core concepts in this document", use_container_width=True):
                st.session_state.pending_query = "Summarize the core concepts in this document."
                st.rerun()
            if st.button("🔍 Explain the primary methodology and definitions", use_container_width=True):
                st.session_state.pending_query = "Explain the primary methodology and definitions."
                st.rerun()

        with col2:
            if st.button("📋 What are the main takeaways and conclusions?", use_container_width=True):
                st.session_state.pending_query = "What are the main takeaways and conclusions?"
                st.rerun()
            if st.button("❓ What key questions does this document answer?", use_container_width=True):
                st.session_state.pending_query = "What key questions does this document answer?"
                st.rerun()

    else:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-icon-wrapper">✨</div>
                <div class="hero-title">Welcome to Learn RAG</div>
                <div class="hero-subtitle">
                    A cloud-native document intelligence assistant. Upload any text-based PDF in the panel above
                    to automatically generate dense vector embeddings, index them to Qdrant Cloud, and start querying.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# B. Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            citations = msg.get("citations", {})
            citation_summary = citations.get("citation_summary", "")

            # Citations Box
            if citation_summary and "No citations" not in citation_summary:
                st.markdown(
                    f"""
                    <div class="citation-box">
                        <div class="citation-header">
                            <span>📄</span>
                            <span>SOURCE EVIDENCE</span>
                        </div>
                        <div class="citation-body">
                            {citation_summary}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Retrieval Diagnostics (Developer Drawer)
            chunks = msg.get("chunks", [])
            if chunks:
                with st.expander(
                    f"🔍 Retrieval Diagnostics ({len(chunks)} passages from Qdrant Cloud)",
                    expanded=st.session_state.show_diagnostics,
                ):
                    for i, ch in enumerate(chunks, 1):
                        pg = ch.get("page", 1)
                        score = ch.get("score")
                        score_label = f"Score: {score:.4f}" if score is not None else "N/A"
                        char_len = len(ch.get("content", ""))

                        st.markdown(
                            f"""
                            <div class="chunk-diagnostic-card">
                                <div class="chunk-diagnostic-header">
                                    <span>Chunk #{i} &bull; Page {pg}</span>
                                    <span class="chunk-badge">{score_label}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.code(ch.get("content", ""), language="text")

# C. Chat Input Composer
chat_placeholder = (
    f"Ask anything about {st.session_state.active_document['filename']}..."
    if st.session_state.active_document
    else "Upload a PDF above to begin asking questions..."
)

user_query = st.chat_input(
    chat_placeholder,
    disabled=st.session_state.active_document is None,
)

if user_query:
    query_to_execute = user_query

# D. Process Query Execution
if query_to_execute and st.session_state.active_document:
    active_doc = st.session_state.active_document

    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": query_to_execute})
    with st.chat_message("user"):
        st.markdown(query_to_execute)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching Qdrant Cloud vectors & synthesizing grounded answer..."):
            try:
                qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                embeddings = get_embedding_model(config.embedding_model_name)

                retrieved_docs = retrieve_relevant_chunks(
                    client=qdrant_client,
                    embeddings_model=embeddings,
                    collection_name=config.qdrant_collection_name,
                    query=query_to_execute,
                    doc_id=active_doc["doc_id"],
                    top_k=config.top_k,
                )

                result = generate_grounded_answer(
                    query=query_to_execute,
                    retrieved_chunks=retrieved_docs,
                    openrouter_api_key=config.openrouter_api_key,
                    model_name=config.openrouter_model,
                )

                answer_text = result["answer"]
                citations = result["citations"]
                citation_summary = citations.get("citation_summary", "")

                st.markdown(answer_text)

                if citation_summary and "No citations" not in citation_summary:
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <div class="citation-header">
                                <span>📄</span>
                                <span>SOURCE EVIDENCE</span>
                            </div>
                            <div class="citation-body">
                                {citation_summary}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                chunk_details = [
                    {
                        "page": doc.metadata.get("page", 1),
                        "score": doc.metadata.get("similarity_score"),
                        "content": doc.page_content,
                    }
                    for doc in retrieved_docs
                ]

                if chunk_details:
                    with st.expander(
                        f"🔍 Retrieval Diagnostics ({len(chunk_details)} passages from Qdrant Cloud)",
                        expanded=st.session_state.show_diagnostics,
                    ):
                        for i, ch in enumerate(chunk_details, 1):
                            pg = ch["page"]
                            sc = ch["score"]
                            sc_label = f"Score: {sc:.4f}" if sc is not None else "N/A"
                            st.markdown(
                                f"""
                                <div class="chunk-diagnostic-card">
                                    <div class="chunk-diagnostic-header">
                                        <span>Chunk #{i} &bull; Page {pg}</span>
                                        <span class="chunk-badge">{sc_label}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.code(ch["content"], language="text")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                        "chunks": chunk_details,
                        "model": result.get("model_used"),
                    }
                )

            except Exception as query_exc:
                err_msg = f"❌ Error retrieving answer: {query_exc}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                        "citations": {},
                        "chunks": [],
                    }
                )

# Composer Footer Note
st.markdown(
    """
    <div class="chat-composer-footer">
        Answers strictly grounded in your PDF vectors via Qdrant Cloud & OpenRouter &bull; Verified Retrieval
    </div>
    """,
    unsafe_allow_html=True,
)
