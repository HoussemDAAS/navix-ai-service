import logging
from fastapi import APIRouter, HTTPException

from schemas.directions import DirectionsRequest, DirectionsResponse
from agents.direction_generator import run_direction_generation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/direction-generation", response_model=DirectionsResponse)
async def generate_directions(request: DirectionsRequest):
    """Generate content directions from analysis results."""
    try:
        result = await run_direction_generation(request)
        return result
    except Exception as e:
        logger.error(f"Direction generation failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
