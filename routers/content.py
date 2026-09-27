import logging
from fastapi import APIRouter, HTTPException

from schemas.content import ContentUnderstandRequest, ContentUnderstandResponse
from agents.content_reader import understand_content

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content", tags=["content"])


@router.post("/understand", response_model=ContentUnderstandResponse)
async def understand(request: ContentUnderstandRequest):
    """Transcribe and describe a batch of scraped posts."""
    try:
        return await understand_content(request)
    except Exception as e:
        logger.error(f"Content understanding failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
