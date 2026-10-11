from pydantic import BaseModel, Field


class KnowledgeTagItem(BaseModel):
    id: str
    type: str
    content: str
    title: str | None = None


class KnowledgeTagRequest(BaseModel):
    items: list[KnowledgeTagItem] = Field(default_factory=list)


class KnowledgeTag(BaseModel):
    id: str
    title: str | None = None
    language: str | None = None
    script: str | None = None
    field: str | None = None
    format: str | None = None
    goal: str | None = None
    funnel: str | None = None
    note: str | None = None


class KnowledgeTagResponse(BaseModel):
    tags: list[KnowledgeTag] = Field(default_factory=list)


class ArticleDistillRequest(BaseModel):
    url: str
    text: str
    field: str | None = None


class DistilledPrinciple(BaseModel):
    type: str  # rule | framework | hook | cta
    title: str
    content: str
    format: str | None = None
    goal: str | None = None


class ArticleDistillResponse(BaseModel):
    title: str = ""
    language: str = "en"
    principles: list[DistilledPrinciple] = Field(default_factory=list)
