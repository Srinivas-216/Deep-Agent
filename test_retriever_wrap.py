import asyncio
from rag_engine import get_retriever, load_pdf, retrieved_docs_var

async def test():
    pdf_path = "uploads/Pinamaraju_Srinivas_Resume.pdf"
    load_pdf(pdf_path)
    
    retrieved_docs_var.set([])
    retriever = get_retriever()
    
    print("Calling retriever.invoke...")
    try:
        docs = retriever.invoke("education")
        print(f"invoke succeeded: retrieved {len(docs)} documents.")
    except Exception as e:
        print(f"invoke failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nCalling retriever.ainvoke...")
    try:
        docs = await retriever.ainvoke("education")
        print(f"ainvoke succeeded: retrieved {len(docs)} documents.")
    except Exception as e:
        print(f"ainvoke failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
