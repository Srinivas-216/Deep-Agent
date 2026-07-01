import asyncio
import os
import contextvars
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.fake import FakeListLLM

# Simulate context variables
active_pdf_var = contextvars.ContextVar("active_pdf_var", default=None)
retrieved_docs_var = contextvars.ContextVar("retrieved_docs_var", default=None)

class MockDoc:
    def __init__(self, page_content, page, source):
        self.page_content = page_content
        self.metadata = {"page": page, "source": source}

@tool
async def search_documents_async(query: str) -> str:
    """Search documents tool."""
    print(f"[Tool] search_documents_async called with: {query}")
    collected = retrieved_docs_var.get()
    print(f"[Tool] collected list: {collected}")
    if collected is not None:
        collected.append(MockDoc("Mock passage for " + query, 3, "test.pdf"))
        print(f"[Tool] Appended to collected list: {collected}")
    else:
        print("[Tool] ERROR: collected list is None!")
    return "Found mock content for " + query

async def main():
    token = active_pdf_var.set("test.pdf")
    docs_token = retrieved_docs_var.set([])

    print(f"[Main] Initial context set. retrieved_docs_var: {retrieved_docs_var.get()}")

    # Fake LLM that will invoke tool call and then respond
    # FakeListLLM takes responses as a list of string prompts / tool calls.
    # To mock a tool call, we can specify the assistant message containing the tool call.
    # Note: FakeListLLM is standard in langchain_core.language_models
    
    # We will just manually test invoking the tool as LangGraph agent would do
    print("\n--- Invoking tool directly via ainvoke ---")
    await search_documents_async.ainvoke({"query": "candidate name"})

    print("\n--- Checking final retrieved_docs_var ---")
    print(f"[Main] Final retrieved_docs_var: {retrieved_docs_var.get()}")
    if retrieved_docs_var.get():
        print(f"[Main] SUCCESS: ContextVar propagated and collected {len(retrieved_docs_var.get())} docs.")
    else:
        print("[Main] FAILURE: ContextVar was not propagated or updated.")

    active_pdf_var.reset(token)
    retrieved_docs_var.reset(docs_token)

if __name__ == "__main__":
    asyncio.run(main())
