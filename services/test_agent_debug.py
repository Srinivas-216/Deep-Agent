import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from rag_engine import load_pdf, active_pdf_var, retrieved_docs_var, set_request_pdf
from agent_engine import agent_executor

async def test():
    pdf_path = "uploads/Pinamaraju_Srinivas_Resume.pdf"
    load_pdf(pdf_path)
    
    active_pdf_var.set("Pinamaraju_Srinivas_Resume.pdf")
    retrieved_docs_var.set([])
    set_request_pdf("Pinamaraju_Srinivas_Resume.pdf")

    question = "Who is the candidate and what is their education?"
    print(f"Querying agent: {question}")
    
    try:
        inputs = {"messages": [("user", question)]}
        async for event in agent_executor.astream_events(inputs, version="v2"):
            event_type = event["event"]
            if event_type == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
            elif event_type == "on_tool_start":
                print(f"\n[Tool Start] {event['name']} with input {event['data'].get('input')}")
            elif event_type == "on_tool_end":
                print(f"\n[Tool End] {event['name']}")
    except Exception as e:
        print("\n\nException occurred:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
