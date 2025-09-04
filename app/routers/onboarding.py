# app/routers/onboarding.py
from fastapi import APIRouter
from app.models import OnboardingInput
from app.services.store import save_profile

router = APIRouter()

@router.post("")
def create_profile(data: OnboardingInput):
    sid = save_profile(data.model_dump())
    return {"session_id": sid}
