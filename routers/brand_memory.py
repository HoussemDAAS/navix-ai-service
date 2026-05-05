import logging
from fastapi import APIRouter, HTTPException

from schemas.brand_memory import BrandMemoryRequest, BrandMemoryResponse
from agents.brand_memory import run_brand_memory_update

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/brand-memory", response_model=BrandMemoryResponse)
async def update_brand_memory(request: BrandMemoryRequest):
    """Update brand memory based on user feedback on drafts."""
    try:
        result = await run_brand_memory_update(request)
        return result
    except Exception as e:
        logger.error(f"Brand memory update failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
