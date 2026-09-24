from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: str | None = None
    text: str
    kind: str = "preference"  # preference | avoid | topic | format | voice | audience | goal
    source: str = ""
    learned_at: str | None = None


class CreatorMemory(BaseModel):
    summary: str = ""
    learned: list[MemoryItem] = Field(default_factory=list)


class TranscriptMessage(BaseModel):
    role: str
    content: str


class FeedbackSignal(BaseModel):
    action: str  # approved | rejected | edited
    caption: str | None = None
    notes: str | None = None
    idea_title: str | None = None


class CreatorMemoryUpdateRequest(BaseModel):
    project_id: str
    persona: str | None = None
    brand_kit: dict | None = None
    current_memory: CreatorMemory = Field(default_factory=CreatorMemory)
    transcript: list[TranscriptMessage] = Field(default_factory=list)
    feedback: list[FeedbackSignal] = Field(default_factory=list)
    session_label: str = "a recent session"


class CreatorMemoryUpdateResponse(BaseModel):
    memory: CreatorMemory
    changed: bool = True
