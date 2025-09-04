# app/routers/chat.py
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.services.generator import answer, answer_stream

router = APIRouter()

class ChatIn(BaseModel):
    session_id: str
    message: str

@router.post("")
def chat(payload: ChatIn):
    reply, sources = answer(payload.session_id, payload.message)
    return JSONResponse({"reply": reply, "sources": sources})

@router.get("/stream")
def chat_stream(session_id: str = Query(...), q: str = Query(...)):
    gen = answer_stream(session_id, q)
    return StreamingResponse(gen, media_type="text/plain")
