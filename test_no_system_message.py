import sys
import os
sys.path.append("d:/my-agent/Backend")

import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from agent_tools import TOOLS

load_dotenv(dotenv_path="d:/my-agent/Backend/.env")

async def test():
    groq_model = os.getenv("GROQ_MODEL")
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=groq_model, max_tokens=512, api_key=groq_api_key)
    
    # Create agent without prompt / state_modifier
    agent = create_react_agent(llm, tools=TOOLS)
    inputs = {"messages": [("user", "What is the training set used in the paper?")]}
    
    try:
        async for event in agent.astream_events(inputs, version="v2"):
            event_type = event["event"]
            if event_type == "on_tool_start":
                print(f"Success! Model called tool: {event['name']} with input: {event['data'].get('input')}")
            elif event_type == "on_chat_model_stream":
                print(event['data']['chunk'].content, end="", flush=True)
    except Exception as e:
        print("\nFailed with error:", e)
        if hasattr(e, 'body'):
            print("Failed generation:", e.body.get('failed_generation'))

if __name__ == "__main__":
    asyncio.run(test())
