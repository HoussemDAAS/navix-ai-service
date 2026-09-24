"""
Creator memory: the durable facts about how one creator wants their content made,
learned from Studio conversations and from which drafts they approved or rejected.
The model sees the current memory and returns the whole updated memory; the
backend keeps item ids stable so the user can delete entries.
"""
import json
import logging
import os

from openai import AsyncOpenAI

from schemas.creator_memory import (
    CreatorMemory,
    CreatorMemoryUpdateRequest,
    CreatorMemoryUpdateResponse,
    MemoryItem,
)

logger = logging.getLogger(__name__)

MEMORY_MODEL = os.getenv("MEMORY_MODEL", "gpt-4o-mini")
MAX_ITEMS = 30
KINDS = {"preference", "avoid", "topic", "format", "voice", "audience", "goal"}

SYSTEM_PROMPT = """You maintain a creator memory: the durable facts about how ONE creator wants their content made, learned from their conversations with a content assistant and from which drafts they approved or rejected.

Rules:
- Keep every prior item unless the new material contradicts it (then replace it). Merge duplicates.
- Add only what the material supports: an explicit request, a repeated choice, an approval or a rejection. Never invent, never generalise from one word.
- The brand kit is already known to every agent: do not copy it into the memory. Record only what the conversations and decisions add beyond it (exceptions, nuances, specific likes and dislikes, recurring themes).
- Each item is one concrete sentence of at most 140 characters, useful to a copywriter. Prefer "wants hooks that state a number in the first line" over "likes good hooks".
- kind is one of: preference, avoid, topic, format, voice, audience, goal.
- source names where it came from (the session label, or "feedback").
- At most 30 items in total; drop the least useful old ones if needed.
- summary: two or three sentences on who this creator is and what they make, updated if the material changes it.
Return ONLY JSON: {"summary": "...", "learned": [{"text": "...", "kind": "...", "source": "..."}]}"""


def _format_memory(memory: CreatorMemory) -> str:
    if not memory.summary and not memory.learned:
        return "(empty — first session)"
    lines = [f"Summary: {memory.summary or '(none yet)'}"]
    for item in memory.learned:
        lines.append(f"- [{item.kind}] {item.text} (source: {item.source or 'unknown'})")
    return "\n".join(lines)


def _format_transcript(request: CreatorMemoryUpdateRequest) -> str:
    lines = []
    for m in request.transcript[-40:]:
        speaker = "USER" if m.role == "user" else "NAVIX"
        lines.append(f"{speaker}: {m.content.strip()[:700]}")
    return "\n".join(lines) or "(no conversation)"


def _format_feedback(request: CreatorMemoryUpdateRequest) -> str:
    if not request.feedback:
        return "(none)"
    lines = []
    for f in request.feedback[-20:]:
        caption = (f.caption or "").replace("\n", " ")[:220]
        title = f" — idea: {f.idea_title}" if f.idea_title else ""
        notes = f" — notes: {f.notes}" if f.notes else ""
        lines.append(f"{f.action.upper()}{title}: \"{caption}\"{notes}")
    return "\n".join(lines)


def _format_brand_kit(kit: dict | None) -> str:
    if not kit:
        return "(not set)"
    return ", ".join(
        f"{key}: {value}"
        for key, value in kit.items()
        if key in ("tone_of_voice", "formality_level", "target_audience", "objective", "preferred_cta") and value
    ) or "(not set)"


async def update_creator_memory(request: CreatorMemoryUpdateRequest) -> CreatorMemoryUpdateResponse:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""ACCOUNT TYPE: {request.persona or 'unknown'}
BRAND KIT: {_format_brand_kit(request.brand_kit)}

CURRENT MEMORY:
{_format_memory(request.current_memory)}

NEW MATERIAL — {request.session_label}:
{_format_transcript(request)}

DRAFT DECISIONS:
{_format_feedback(request)}

Return the full updated memory as JSON."""

    response = await client.chat.completions.create(
        model=MEMORY_MODEL,
        temperature=0.2,
        max_tokens=1800,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "{}").strip()
    parsed = json.loads(raw)

    items: list[MemoryItem] = []
    seen: set[str] = set()
    for entry in parsed.get("learned") or []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        kind = str(entry.get("kind") or "preference").strip().lower()
        items.append(
            MemoryItem(
                text=text[:160],
                kind=kind if kind in KINDS else "preference",
                source=str(entry.get("source") or request.session_label)[:80],
            )
        )
        if len(items) >= MAX_ITEMS:
            break

    memory = CreatorMemory(summary=str(parsed.get("summary") or request.current_memory.summary or "")[:600], learned=items)
    changed = memory.summary != request.current_memory.summary or [i.text for i in items] != [
        i.text for i in request.current_memory.learned
    ]
    logger.info(f"Creator memory for project {request.project_id}: {len(items)} items (changed={changed})")
    return CreatorMemoryUpdateResponse(memory=memory, changed=changed)
