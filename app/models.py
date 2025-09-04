# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class OnboardingInput(BaseModel):
    company_size: Optional[str] = None
    industry: Optional[str] = None
    state: Optional[str] = None
    categories: List[str] = []

class ChatInput(BaseModel):
    session_id: str
    message: str

class SearchHit(BaseModel):
    bill_id: str
    title: str
    state: str
    category: str
    date: Optional[str] = None
    url: Optional[str] = None
    score: float
