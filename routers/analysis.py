import logging
from fastapi import APIRouter, HTTPException

from schemas.analysis import AnalysisRequest, AnalysisResponse
from agents.market_analysis import run_market_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/market-analysis", response_model=AnalysisResponse)
async def market_analysis(request: AnalysisRequest):
    """Run market analysis on competitor data and signals."""
    try:
        result = await run_market_analysis(request)
        return result
    except Exception as e:
        logger.error(f"Market analysis failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
