from pydantic import BaseModel, Field


class BrandKitProfile(BaseModel):
    name: str | None = None
    bio: str | None = None
    niche: str | None = None


class BrandKitPost(BaseModel):
    caption: str
    likes: int | None = None
    comments: int | None = None
    views: int | None = None
    format: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    published_at: str | None = None


class BrandKitDeriveRequest(BaseModel):
    project_id: str
    persona: str | None = None  # creator | ecommerce | agency
    profile: BrandKitProfile
    posts: list[BrandKitPost]
    # The UI offers fixed tiles; the model must answer with one of these
    tone_options: list[str]
    formality_options: list[str]


class BrandKitDeriveResponse(BaseModel):
    tone_of_voice: str | None = None
    formality_level: str | None = None
    target_audience: str | None = None
    objective: str | None = None
    preferred_cta: str | None = None
    vocab_exclude: str | None = None
    constraints: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
