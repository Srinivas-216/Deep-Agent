import asyncio
import os
import contextvars
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.fake import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

# Simulate our context variables
active_pdf_var = contextvars.ContextVar("active_pdf_var", default=None)
retrieved_docs_var = contextvars.ContextVar("retrieved_docs_var", default=None)

class MockDoc:
    def __init__(self, page_content, page, source):
        self.page_content = page_content
        self.metadata = {"page": page, "source": source}

@tool
def search_documents(query: str) -> str:
    """Search documents tool."""
    print(f"[Tool] search_documents called with: {query}")
    # Get current context value
    collected = retrieved_docs_var.get()
    print(f"[Tool] collected list before append: {collected}")
    if collected is not None:
        collected.append(MockDoc("Mock passage for " + query, 3, "test.pdf"))
        print(f"[Tool] Appended to collected list: {collected}")
    else:
        print("[Tool] ERROR: collected list is None (context not propagated)!")
    return "Found mock content for " + query

async def main():
    # Set context vars
    token = active_pdf_var.set("test.pdf")
    docs_token = retrieved_docs_var.set([])

    print(f"[Main] Initial context set. retrieved_docs_var: {retrieved_docs_var.get()}")

    # Setup fake agent to trigger tool call
    # The agent will decide to call search_documents
    mock_llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_documents", "args": {"query": "candidate name"}, "id": "call_1"}]
            ),
            AIMessage(content="The candidate is John Doe (Page 4).")
        ]
    )
    
    agent = create_react_agent(mock_llm, tools=[search_documents])
    
    print("\n--- Running agent ---")
    inputs = {"messages": [("user", "What is the candidate's name?")]}
    async for event in agent.astream_events(inputs, version="v2"):
        event_type = event["event"]
        if event_type == "on_chat_model_stream":
            print(f"Token: {event['data']['chunk'].content}")
        elif event_type == "on_tool_start":
            print(f"Tool start: {event['name']}")
        elif event_type == "on_tool_end":
            print(f"Tool end: {event['name']}")

    print("\n--- Finished running agent ---")
    print(f"[Main] Final retrieved_docs_var: {retrieved_docs_var.get()}")
    if retrieved_docs_var.get():
        print(f"[Main] SUCCESS: Collected {len(retrieved_docs_var.get())} docs.")
    else:
        print("[Main] FAILURE: Collected 0 docs.")

    active_pdf_var.reset(token)
    retrieved_docs_var.reset(docs_token)

if __name__ == "__main__":
    asyncio.run(main())
