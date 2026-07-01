import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import AsyncGenerator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

import contextvars
import threading

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

active_pdf_var = contextvars.ContextVar("active_pdf_var", default=None)


_thread_local = threading.local()


vector_stores: dict[str, FAISS] = {}
pdf_metadata: dict[str, dict] = {}
_current_pdf: str | None = None
UPLOAD_DIR = "uploads"


def set_request_pdf(filename: str | None):
    """Pin the active PDF for the current OS thread (used by agent tools)."""
    _thread_local.active_pdf = filename


def get_active_filename() -> str | None:
    """
    Resolve the active PDF filename with priority:
    1. ContextVar (set at request boundary)
    2. Thread-local (set for agent tool calls)
    3. Global _current_pdf (last uploaded)

    NOTE: Returns None when no PDF is explicitly pinned — callers that want
    cross-PDF auto-search should check for None and use search_all_pdfs()
    instead of falling back to _current_pdf silently.
    """
    return active_pdf_var.get() or getattr(_thread_local, "active_pdf", None)


def get_pinned_filename() -> str | None:
    """Strict version — only returns an explicitly pinned filename, never the global default."""
    return active_pdf_var.get() or getattr(_thread_local, "active_pdf", None)


def get_pdf_metadata(filename: str | None = None) -> dict | None:
    fn = filename or get_current_pdf()
    if not fn:
        return None
    if fn not in pdf_metadata:
        file_path = os.path.join(UPLOAD_DIR, fn)
        if os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                pdf_metadata[fn] = {
                    "page_count": len(pages),
                    "file_size": os.path.getsize(file_path)
                }
            except Exception:
                return {"name": fn, "page_count": "Unknown", "file_size": os.path.getsize(file_path)}
        else:
            return None
    meta = pdf_metadata.get(fn, {})
    return {
        "name": fn,
        "page_count": meta.get("page_count", "Unknown"),
        "file_size": meta.get("file_size", 0)
    }


def load_pdf(file_path: str) -> int:
    global _current_pdf

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    _current_pdf = filename

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    pdf_metadata[filename] = {
        "page_count": len(pages),
        "file_size": os.path.getsize(file_path)
    }

    for page in pages:
        page.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No text content could be extracted from PDF: {filename}")
    store = FAISS.from_documents(chunks, embeddings)

    vector_stores[filename] = store
    return len(chunks)


def init_existing_pdfs():
    """Index all PDFs currently in the uploads directory on startup."""
    global _current_pdf
    if not os.path.exists(UPLOAD_DIR):
        return
    for filename in os.listdir(UPLOAD_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                print(f"Auto-indexing existing document on startup: {filename}")
                load_pdf(file_path)
                _current_pdf = filename
            except Exception as e:
                print(f"Failed to auto-index {filename}: {e}")


def get_available_pdfs() -> list[dict]:
    """Return metadata for all PDFs currently present in the uploads directory."""
    pdfs = []
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(UPLOAD_DIR, filename)
                size = os.path.getsize(file_path)
                store = vector_stores.get(filename)
                chunks = store.index.ntotal if store else 0
                pdfs.append({
                    "name": filename,
                    "size": size,
                    "chunks": chunks
                })
    return pdfs


def get_current_pdf() -> str | None:
    """Used for display/status purposes — falls back to last-loaded PDF."""
    return get_active_filename() or _current_pdf


# Context var to collect retrieved docs for citation output
retrieved_docs_var = contextvars.ContextVar("retrieved_docs_var", default=None)


def get_retriever(k: int = 4, filename: str | None = None):
    """Return a retriever from a SPECIFIC vector store (single PDF)."""
    filename = filename or get_active_filename()
    if not filename:
        raise ValueError("No PDF has been uploaded yet.")

    if filename not in vector_stores:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            load_pdf(file_path)
        else:
            raise ValueError(f"Selected PDF '{filename}' is not indexed and not found on disk.")

    retriever = vector_stores[filename].as_retriever(search_kwargs={"k": k})

    original_invoke = retriever.invoke
    def wrapped_invoke(*args, **kwargs):
        docs = original_invoke(*args, **kwargs)
        collected = retrieved_docs_var.get()
        if collected is not None:
            collected.extend(docs)
        return docs
    retriever.__dict__["invoke"] = wrapped_invoke

    original_ainvoke = retriever.ainvoke
    async def wrapped_ainvoke(*args, **kwargs):
        docs = await original_ainvoke(*args, **kwargs)
        collected = retrieved_docs_var.get()
        if collected is not None:
            collected.extend(docs)
        return docs
    retriever.__dict__["ainvoke"] = wrapped_ainvoke

    return retriever


# ── Cross-PDF auto-search ─────────────────────────────────────────────────────

async def search_all_pdfs(query: str, k_per_pdf: int = 4):
    """
    Search every indexed PDF for the query, rank by FAISS similarity score,
    and return results from the SINGLE best-matching PDF only.

    Returns: (best_filename, docs) or (None, []) if nothing is indexed.
    """
    if not vector_stores:
        return None, []

    best_filename = None
    best_score = float("inf")  # FAISS L2 distance — lower is better
    best_docs = []

    for filename, store in vector_stores.items():
        try:
            # similarity_search_with_score returns (doc, distance) — lower distance = better match
            results = store.similarity_search_with_score(query, k=k_per_pdf)
        except Exception:
            continue

        if not results:
            continue

        top_score = results[0][1]  # best (lowest) distance for this PDF
        if top_score < best_score:
            best_score = top_score
            best_filename = filename
            best_docs = [doc for doc, score in results]

    # Track for citations
    collected = retrieved_docs_var.get()
    if collected is not None and best_docs:
        collected.extend(best_docs)

    return best_filename, best_docs

def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512,
        temperature=0.0
    )


def format_docs(docs) -> str:
    formatted = []
    for i, d in enumerate(docs):
        source = d.metadata.get('source', '?')
        page = int(d.metadata.get('page', 0)) + 1
        formatted.append(
            f"Document [{i+1}] (Source: {source}, Page: {page}):\n{d.page_content}"
        )
    return "\n\n".join(formatted)


def get_chat_prompt(template_str: str) -> ChatPromptTemplate:
    clean_template = template_str.replace("\\n", "\n")
    if "Context:" in clean_template:
        parts = clean_template.split("Context:", 1)
        system_instruction = parts[0].strip()
        human_template = "Context:\n" + parts[1].strip()
        return ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", human_template)
        ])
    return ChatPromptTemplate.from_template(clean_template)


def get_rag_chain(filename: str | None = None):
    """Simple LCEL chain — kept for agent tools that need a single-PDF chain."""
    filename = filename or get_active_filename()
    if filename is None:
        raise ValueError("No PDF has been uploaded yet.")

    retriever = get_retriever(filename=filename)

    prompt_template = os.getenv("RAG_PROMPT_TEMPLATE")
    if not prompt_template:
        raise ValueError("RAG_PROMPT_TEMPLATE is not set in the .env file")

    RAG_PROMPT = get_chat_prompt(prompt_template)
    llm = get_llm()

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def retrieve_with_citations(question: str) -> tuple[list, str]:
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = format_docs(docs)
    return docs, context