from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models.schemas import QARequest, QAResponse
from app.services.qa_pipeline import run_qa

router = APIRouter(prefix="/api/v1", tags=["qa"])


@router.post("/qa", response_model=QAResponse)
async def post_qa(
    body: QARequest,
    settings: Settings = Depends(get_settings),
) -> QAResponse:
    return await run_qa(settings, body)
