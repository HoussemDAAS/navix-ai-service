from pydantic import BaseModel
from .common import BrandKitData
from .directions import ContentDirection


class Draft(BaseModel):
    direction_title: str
    caption_text: str
    video_script: str | None = None
    hook_alternatives: list[str]
    cta_alternatives: list[str]


class DraftsRequest(BaseModel):
    project_id: str
    directions: list[ContentDirection]
    brand_kit: BrandKitData


class DraftsResponse(BaseModel):
    drafts: list[Draft]
