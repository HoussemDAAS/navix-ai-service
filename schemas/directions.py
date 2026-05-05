from pydantic import BaseModel
from .common import BrandKitData
from .analysis import AnalysisResponse


class ContentDirection(BaseModel):
    title_pillar: str
    angle: str
    format: str
    hook_suggestion: str
    cta_suggestion: str
    rationale: str


class DirectionsRequest(BaseModel):
    project_id: str
    analysis: AnalysisResponse
    brand_kit: BrandKitData
    num_directions: int = 6


class DirectionsResponse(BaseModel):
    directions: list[ContentDirection]
