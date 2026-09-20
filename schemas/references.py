from pydantic import BaseModel, Field


class ReferenceExtractRequest(BaseModel):
    reference_id: str
    kind: str  # image | pdf | text | video | audio | link
    url: str  # public URL of the uploaded file (or the link itself)
    mime_type: str | None = None
    title: str | None = None


class ReferenceExtractResponse(BaseModel):
    extracted_text: str = ""
    summary: str = ""
    meta: dict = Field(default_factory=dict)
