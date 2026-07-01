import asyncio
import os
from rag_engine import load_pdf, stream_rag_response, get_current_pdf, get_active_filename

async def main():
    pdf_path = "uploads/Pinamaraju_Srinivas_Resume.pdf"
    print(f"Loading PDF: {pdf_path}")
    try:
        chunks = load_pdf(pdf_path)
        print(f"Successfully loaded PDF, chunks: {chunks}")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        import traceback
        traceback.print_exc()
        return

    question = "What is the candidate's name and experience?"
    print(f"\nQuerying: {question}")
    try:
        async for event in stream_rag_response(question, filename="Pinamaraju_Srinivas_Resume.pdf"):
            print(event)
    except Exception as e:
        print(f"Error streaming response: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
