# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from dotenv import load_dotenv
from app.routers import onboarding, search, chat

load_dotenv()
app = FastAPI(title="AI Legislation Bot")

# keep permissive in dev; tighten for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

# serve the frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
