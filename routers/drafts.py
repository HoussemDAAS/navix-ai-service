import logging
from fastapi import APIRouter, HTTPException

from schemas.drafts import DraftsRequest, DraftsResponse
from agents.draft_generator import run_draft_generation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/draft-generation", response_model=DraftsResponse)
async def generate_drafts(request: DraftsRequest):
    """Generate content drafts from selected directions."""
    try:
        result = await run_draft_generation(request)
        return result
    except Exception as e:
        logger.error(f"Draft generation failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
