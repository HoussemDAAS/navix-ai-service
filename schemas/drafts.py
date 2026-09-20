from typing import Any

from pydantic import BaseModel, field_validator
from .common import BrandKitData
from .directions import ContentDirection


def script_to_text(value: Any) -> str | None:
    """The model sometimes answers a shot-by-shot plan as a list of objects; the
    editor stores one string, so render such plans as numbered lines."""
    if value is None or isinstance(value, str):
        return value
    items = [value] if isinstance(value, dict) else value if isinstance(value, list) else [str(value)]
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        if isinstance(item, dict):
            title = item.get("shot") or item.get("scene") or item.get("title") or f"Shot {index}"
            details = "; ".join(
                f"{key}: {val}" for key, val in item.items()
                if key not in ("shot", "scene", "title") and val
            )
            lines.append(f"{index}. {title}" + (f" — {details}" if details else ""))
        else:
            lines.append(f"{index}. {item}")
    return "\n".join(lines)


class Draft(BaseModel):
    direction_title: str
    caption_text: str
    video_script: str | None = None
    hook_alternatives: list[str]
    cta_alternatives: list[str]

    @field_validator("video_script", mode="before")
    @classmethod
    def _coerce_video_script(cls, value: Any) -> str | None:
        return script_to_text(value)

    @field_validator("hook_alternatives", "cta_alternatives", mode="before")
    @classmethod
    def _coerce_alternatives(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value]


class DraftsRequest(BaseModel):
    project_id: str
    directions: list[ContentDirection]
    brand_kit: BrandKitData
    persona: str | None = None
    account_model: dict | None = None


class DraftsResponse(BaseModel):
    drafts: list[Draft]
