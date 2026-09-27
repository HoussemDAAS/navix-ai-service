from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    id: str
    platform: str = "instagram"
    content_type: str | None = None
    caption: str | None = None
    # Media the backend found in the scraped row; links expire, so this runs right after the scrape
    video_url: str | None = None
    audio_url: str | None = None
    thumbnail_url: str | None = None
    duration_seconds: float | None = None
    image_count: int | None = None


class ContentUnderstandRequest(BaseModel):
    project_id: str
    items: list[ContentItem] = Field(default_factory=list)


class ContentUnderstanding(BaseModel):
    id: str
    transcript: str | None = None
    visual_summary: str | None = None
    language: str | None = None
    error: str | None = None


class ContentUnderstandResponse(BaseModel):
    items: list[ContentUnderstanding] = Field(default_factory=list)
