import logging
from fastapi import APIRouter, HTTPException

from schemas.brand_kit import BrandKitDeriveRequest, BrandKitDeriveResponse
from agents.brand_kit_deriver import derive_brand_kit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/brand-kit-derive", response_model=BrandKitDeriveResponse)
async def brand_kit_derive(request: BrandKitDeriveRequest):
    """Derive a brand kit from the account's own posts."""
    try:
        return await derive_brand_kit(request)
    except Exception as e:
        logger.error(f"Brand kit derivation failed for project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
