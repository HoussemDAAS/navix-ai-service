from pydantic import BaseModel
from .common import BrandKitData, FeedbackData


class BrandMemoryRequest(BaseModel):
    project_id: str
    feedback: list[FeedbackData]
    current_brand_kit: BrandKitData


class BrandMemoryResponse(BaseModel):
    updated_preferences: dict
    learned_patterns: list[str]
