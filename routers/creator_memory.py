import logging
from fastapi import APIRouter, HTTPException

from schemas.creator_memory import CreatorMemoryUpdateRequest, CreatorMemoryUpdateResponse
from agents.creator_memory import update_creator_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/creator-memory", response_model=CreatorMemoryUpdateResponse)
async def creator_memory(request: CreatorMemoryUpdateRequest):
    """Fold a session's conversation and draft decisions into the creator memory."""
    try:
        return await update_creator_memory(request)
    except Exception as e:
        logger.error(f"Creator memory update failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
