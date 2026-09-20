import logging
from fastapi import APIRouter, HTTPException

from schemas.references import ReferenceExtractRequest, ReferenceExtractResponse
from agents.reference_reader import extract_reference

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/references", tags=["references"])


@router.post("/extract", response_model=ReferenceExtractResponse)
async def extract(request: ReferenceExtractRequest):
    """Extract readable text and a brief from a reference the user shared."""
    try:
        return await extract_reference(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Reference extraction failed for {request.reference_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
