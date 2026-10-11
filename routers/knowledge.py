import logging
from fastapi import APIRouter, HTTPException

from schemas.knowledge import (
    ArticleDistillRequest,
    ArticleDistillResponse,
    KnowledgeTagRequest,
    KnowledgeTagResponse,
)
from agents.knowledge import distill_article, tag_items

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/tag", response_model=KnowledgeTagResponse)
async def tag(request: KnowledgeTagRequest):
    """Suggest language, script, field, format, goal and funnel tags for library items."""
    try:
        return await tag_items(request.items)
    except Exception as e:
        logger.error(f"Knowledge tagging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/distill", response_model=ArticleDistillResponse)
async def distill(request: ArticleDistillRequest):
    """Turn an article into standalone, reusable principles for the library."""
    try:
        return await distill_article(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Article distilling failed for {request.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
