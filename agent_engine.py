import os
from collections import Counter
from typing import AsyncGenerator

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

from rag_engine import (
    get_llm,
    active_pdf_var,
    retrieved_docs_var,
    set_request_pdf,
)
from agent_tools import TOOLS

SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    # Fallback if .env key is missing
    "You are a helpful PDF document assistant. "
    "Call the search_documents tool ONCE to find relevant information. "
    "Always mention the page number like: (Page 5). "
    "If the information is not found, clearly say so."
)

def build_agent():
    return create_react_agent(
        model=get_llm(),
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )

agent = build_agent()


def reset_agent():
    """Rebuild the agent after a new PDF is uploaded."""
    global agent
    agent = build_agent()


def _is_valid_input(text: str) -> bool:
    """
    Fast Python-only gibberish check.
    Catches keyboard mashing without any LLM call — zero latency.
    """
    t = text.strip()
    if len(t) < 3:
        return False
    alpha = [c for c in t if c.isalpha()]
    if not alpha:
        return False
    alpha_str = "".join(alpha).lower()
    
    if Counter(alpha_str).most_common(1)[0][1] / len(alpha_str) > 0.5:
        return False

    max_run = current_run = 1
    for i in range(1, len(alpha_str)):
        current_run = current_run + 1 if alpha_str[i] == alpha_str[i-1] else 1
        max_run = max(max_run, current_run)
    if max_run > 4:
        return False
    
    if not any(c in "aeiou" for c in alpha_str):
        return False
    return True



async def stream_agent_response(
    question: str, filename: str | None = None
) -> AsyncGenerator[dict, None]:
    """
    Validate input, run the Deep Agent, stream tokens back to frontend.
    """
    
    if not _is_valid_input(question):
        yield {"type": "token", "data": "Please ask a clear question about the PDF."}
        yield {"type": "done"}
        return


    token = active_pdf_var.set(filename)
    docs_token = retrieved_docs_var.set([])
    set_request_pdf(filename)

    try:
        inputs = {"messages": [{"role": "user", "content": question}]}

       
        async for event in agent.astream_events(inputs, version="v2"):
            event_type = event["event"]

            
            if event_type == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield {"type": "token", "data": content}

       
        collected_docs = retrieved_docs_var.get() or []
        seen = set()
        citations = []
        for doc in collected_docs:
            page = int(doc.metadata.get("page", 0)) + 1
            source = doc.metadata.get("source", "?")
            key = f"{source}::{page}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    "page": page,
                    "source": source,
                    "snippet": doc.page_content[:200].replace("\n", " "),
                })

        if citations:
            yield {"type": "citation", "data": citations}

        yield {"type": "done"}

    except Exception as e:
        yield {"type": "error", "data": str(e)}
        yield {"type": "done"}

    finally:
        active_pdf_var.reset(token)
        retrieved_docs_var.reset(docs_token)
        set_request_pdf(None)