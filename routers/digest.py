import logging
from fastapi import APIRouter, HTTPException

from schemas.digest import WeeklyDigestRequest, WeeklyDigestResponse
from agents.digest import write_weekly_digest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/weekly-digest", response_model=WeeklyDigestResponse)
async def weekly_digest(request: WeeklyDigestRequest):
    """One headline and one grounded idea for the Home page, from the week's computed facts."""
    try:
        return await write_weekly_digest(request)
    except Exception as e:
        logger.error(f"Weekly digest failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
