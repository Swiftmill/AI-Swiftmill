from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.rag import RAGPipeline
from core.retriever import KNOWLEDGE_DIR, SUPPORTED_TYPES, KnowledgeBase

load_dotenv()

app = FastAPI(title="AI Local Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

knowledge_base = KnowledgeBase()
rag_pipeline = RAGPipeline(knowledge_base)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user: Optional[str] = Field(default=None)
    allow_web: bool = Field(default=True, alias="allow_web")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message vide.")

    answer = rag_pipeline.answer(
        question=message,
        user_id=payload.user,
        allow_web=payload.allow_web,
    )
    return ChatResponse(answer=answer.text, sources=answer.sources)


@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")

    saved_paths: List[Path] = []
    for upload in files:
        filename = upload.filename or "document"
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_TYPES:
            raise HTTPException(status_code=400, detail=f"Type de fichier non supporté: {extension}")

        safe_name = KnowledgeBase.normalise_filename(filename)
        destination = KNOWLEDGE_DIR / safe_name
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(KNOWLEDGE_DIR)) as tmp:
            temp_path = Path(tmp.name)
            size = 0
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    tmp.close()
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Fichier trop volumineux (50MB max)")
                tmp.write(chunk)

        os.replace(temp_path, destination)
        saved_paths.append(destination)
        await upload.close()

    chunks_added = knowledge_base.ingest_paths(saved_paths)
    return {"status": "ok", "files": [path.name for path in saved_paths], "chunks": chunks_added}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
