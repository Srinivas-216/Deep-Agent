import os
import json
import shutil

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from rag_engine import (
    load_pdf,
    get_current_pdf,
    get_llm,
    init_existing_pdfs,
    get_available_pdfs,
    active_pdf_var,
)
from agent_engine import stream_agent_response, reset_agent

load_dotenv()

app = FastAPI()

# Auto-index existing documents on startup
init_existing_pdfs()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

# Serve uploaded PDFs as static files so the frontend PDF viewer can fetch them
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ── Static UI ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "running", "current_pdf": get_current_pdf()}


# ── PDF Upload ───────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = load_pdf(file_path)
        reset_agent()

        return {
            "message": f"'{file.filename}' uploaded and indexed successfully.",
            "chunks": chunks,
            "filename": file.filename,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/documents")
def list_documents():
    try:
        documents = get_available_pdfs()
        return {"documents": documents}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "File not found."})

        os.remove(file_path)

        from rag_engine import vector_stores, pdf_metadata
        import rag_engine
        vector_stores.pop(filename, None)
        pdf_metadata.pop(filename, None)

        if rag_engine._current_pdf == filename:
            remaining = list(vector_stores.keys())
            rag_engine._current_pdf = remaining[-1] if remaining else None

        reset_agent()
        return {"message": f"'{filename}' deleted successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── WebSocket: /ws/chat — Deep Agent only ────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            filename = None
            try:
                payload = json.loads(raw)
                question = payload.get("question", "").strip()
                filename = payload.get("filename", None)
            except Exception:
                question = raw.strip()

            if not question:
                await websocket.send_text(json.dumps({"type": "error", "data": "Empty question."}))
                continue

            if get_current_pdf() is None and filename is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": "No PDF uploaded. Please upload a PDF first."
                }))
                continue

            async for event in stream_agent_response(question, filename=filename):
                await websocket.send_text(json.dumps(event))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "data": str(e)}))
        except Exception:
            pass