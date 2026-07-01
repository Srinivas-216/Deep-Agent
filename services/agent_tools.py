import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool
from rag_engine import (
    get_retriever,
    format_docs,
    get_pinned_filename,
    search_all_pdfs,
    get_pdf_metadata,
)


@tool
async def search_documents(query: str) -> str:
    """Search the uploaded PDF for passages relevant to the query.
    If a specific PDF is selected searches only that PDF.
    Otherwise automatically searches across all uploaded PDFs
    and returns results from the best matching document."""

    pinned = get_pinned_filename()
    if pinned:
        retriever = get_retriever(k=2, filename=pinned)
        docs = await retriever.ainvoke(query)
        if not docs:
            return f"No relevant passages found in {pinned}."
        return format_docs(docs)

    best_filename, docs = await search_all_pdfs(query, k_per_pdf=2)
    if not docs:
        return "No relevant passages found in any uploaded document."
    return f"[Best matching document: {best_filename}]\n\n" + format_docs(docs)


@tool
async def list_sources(query: str) -> str:
    """List the source document pages most relevant to a query.
    Returns unique filename and page number entries so you know
    which pages contain information about the topic."""

    pinned = get_pinned_filename()
    if pinned:
        retriever = get_retriever(k=3, filename=pinned)
        docs = await retriever.ainvoke(query)
    else:
        _, docs = await search_all_pdfs(query, k_per_pdf=3)

    if not docs:
        return "No documents loaded yet."
    sources = sorted(set(
        f"{d.metadata.get('source', '?')} — Page {d.metadata.get('page', '?')}"
        for d in docs
    ))
    return "\n".join(sources)


@tool
async def get_document_info(query: str = "") -> str:
    """Get metadata about the currently uploaded PDF document such as
    filename total page count and file size. Use this when the user
    asks about the document itself like how many pages it has."""

    meta = get_pdf_metadata()
    if not meta:
        return "No PDF document has been uploaded yet."
    return (
        f"Document Info:\n"
        f"- Filename: {meta['name']}\n"
        f"- Total Pages: {meta['page_count']}\n"
        f"- File Size: {meta['file_size']} bytes"
    )


# Only 3 fast tools — no extra LLM calls inside tools
TOOLS = [search_documents, list_sources, get_document_info]