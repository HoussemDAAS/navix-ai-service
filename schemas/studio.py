from pydantic import BaseModel, Field


class StudioMessage(BaseModel):
    role: str  # user | assistant
    content: str


class StudioChatRequest(BaseModel):
    project_id: str
    persona: str | None = None
    account_model: dict | None = None
    brand_kit: dict | None = None
    project: dict | None = None  # name, niche, handle, location
    creator_memory: str | None = None
    # Language the CONTENT is written in; the assistant itself talks in English
    content_language: str = "same as my posts"
    # References shared in this session: {id, kind, title, summary}
    references: list[dict] = Field(default_factory=list)
    messages: list[StudioMessage] = Field(default_factory=list)
