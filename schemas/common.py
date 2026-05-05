from pydantic import BaseModel


class CompetitorData(BaseModel):
    handle: str
    platform: str
    full_name: str | None = None
    biography: str | None = None
    followers_count: int | None = None
    posts_count: int | None = None


class SignalData(BaseModel):
    competitor_handle: str
    platform: str
    caption: str | None = None
    likes_count: int | None = None
    views_count: int | None = None
    published_at: str | None = None


class BrandKitData(BaseModel):
    tone_of_voice: str | None = None
    formality_level: str | None = None
    target_audience: str | None = None
    objective: str | None = None
    vocab_exclude: str | None = None
    constraints: list[str] | None = None
    preferred_cta: str | None = None


class FeedbackData(BaseModel):
    draft_id: str
    action: str  # approved, rejected, edited
    notes: str | None = None
    caption_text: str | None = None
