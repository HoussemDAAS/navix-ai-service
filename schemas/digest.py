from pydantic import BaseModel, Field


class DigestProject(BaseModel):
    name: str | None = None
    niche: str | None = None
    handle: str | None = None
    language: str | None = None


class WeeklyDigestRequest(BaseModel):
    project_id: str
    persona: str | None = None
    account_model: dict | None = None
    creator_memory: str | None = None
    brand_kit: dict | None = None
    project: DigestProject = Field(default_factory=DigestProject)
    # The computed digest facts (account deltas, own best post, field breakout, pipeline)
    facts: dict = Field(default_factory=dict)


class DigestIdea(BaseModel):
    title: str
    why: str
    format: str | None = None
    studio_prompt: str


class WeeklyDigestResponse(BaseModel):
    headline: str | None = None
    idea: DigestIdea | None = None
