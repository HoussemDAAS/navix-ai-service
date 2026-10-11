"""
Knowledge library helpers: tag imported items and distill articles into reusable
principles. Both are small JSON-mode calls on a cheap model; the library itself is
retrieved by the other agents through rag.build_library_block.
"""
import asyncio
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

from schemas.knowledge import (
    ArticleDistillRequest,
    ArticleDistillResponse,
    DistilledPrinciple,
    KnowledgeTag,
    KnowledgeTagItem,
    KnowledgeTagResponse,
)

logger = logging.getLogger(__name__)

MODEL = os.getenv("KNOWLEDGE_MODEL", "gpt-4o-mini")
BATCH = 25

LANGUAGES = {"derja", "fr", "en", "mixed", "other"}
SCRIPTS = {"arabic", "latin", "mixed", "na"}
FORMATS = {"reel", "carousel", "image", "story", "caption", "any"}
GOALS = {"sell", "educate", "entertain", "announce", "engage", "any"}
FUNNELS = {"tof", "mof", "bof", "any"}
PRINCIPLE_TYPES = {"rule", "framework", "hook", "cta"}

_client: AsyncOpenAI | None = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


TAG_SYSTEM = """You label items of a content-writing library used for Tunisian social media (Instagram, TikTok).
Answer with JSON only. For each item return: language, script, field, format, goal, funnel, title, note.
- language: "derja" for Tunisian dialect (Arabic letters, or Latin letters with digits such as 3, 7, 9), "fr" for French, "en" for English, "mixed" when Derja and French or English alternate inside the item, "other" otherwise. Derja is not Egyptian, Gulf or standard Arabic: words like "برشا", "توا", "باهي", "كيفاش", "شنوة", "هاو", "كان" (if) mark it.
- script: "arabic" (Arabic letters), "latin" (Latin letters), "mixed" (both), "na" (a structure or a rule without natural language).
- field: the niche or industry the item fits, 1 to 3 lowercase words ("e-commerce", "fitness", "car accessories", "coaching"), or null when generic.
- format: reel | carousel | image | story | caption | any.
- goal: sell | educate | entertain | announce | engage | any.
- funnel: tof (cold audience) | mof (already interested) | bof (ready to buy) | any.
- title: at most 60 characters, in the item's own language, describing the item.
- note: one short sentence on why it works or when to use it, in English; null when obvious.
Return {"tags": [{"id": "...", "language": "...", "script": "...", "field": ..., "format": "...", "goal": "...", "funnel": "...", "title": "...", "note": ...}]} covering every id exactly once."""


def _pick(value: Any, allowed: set[str]) -> str | None:
    v = str(value or "").strip().lower()
    return v if v in allowed else None


CONCURRENCY = 6


async def _tag_batch(client: AsyncOpenAI, batch: list[KnowledgeTagItem], sem: asyncio.Semaphore) -> dict[str, dict[str, Any]]:
    payload = [{"id": it.id, "type": it.type, "title": it.title, "content": it.content[:1500]} for it in batch]
    async with sem:
        try:
            completion = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": TAG_SYSTEM},
                    {"role": "user", "content": json.dumps({"items": payload}, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(completion.choices[0].message.content or "{}")
        except Exception as exc:
            logger.warning(f"Knowledge tagging failed for a batch of {len(batch)}: {exc}")
            data = {}
    return {str(t.get("id")): t for t in (data.get("tags") or []) if isinstance(t, dict)}


async def tag_items(items: list[KnowledgeTagItem]) -> KnowledgeTagResponse:
    """Tag in batches of 25, several batches in flight: a 2 000-line import stays under two minutes."""
    tags: list[KnowledgeTag] = []
    client = _openai()
    sem = asyncio.Semaphore(CONCURRENCY)
    batches = [items[start : start + BATCH] for start in range(0, len(items), BATCH)]
    results = await asyncio.gather(*(_tag_batch(client, b, sem) for b in batches))
    for batch, by_id in zip(batches, results):
        for it in batch:
            t = by_id.get(it.id, {})
            field = str(t.get("field") or "").strip().lower()[:60] or None
            tags.append(
                KnowledgeTag(
                    id=it.id,
                    title=(str(t.get("title") or "").strip()[:80] or None),
                    language=_pick(t.get("language"), LANGUAGES),
                    script=_pick(t.get("script"), SCRIPTS),
                    field=field,
                    format=_pick(t.get("format"), FORMATS),
                    goal=_pick(t.get("goal"), GOALS),
                    funnel=_pick(t.get("funnel"), FUNNELS),
                    note=(str(t.get("note") or "").strip()[:300] or None),
                )
            )
    return KnowledgeTagResponse(tags=tags)


DISTILL_SYSTEM = """You read an article about copywriting, social media content or marketing and extract what a copywriter can reuse.
Return JSON: {"title": "...", "language": "en|fr|derja|mixed|other", "principles": [{"type": "rule|framework|hook|cta", "title": "...", "content": "...", "format": "reel|carousel|image|story|caption|any", "goal": "sell|educate|entertain|announce|engage|any"}]}
- 3 to 10 principles. Each one stands alone and is actionable: what to do, when, and why. At most 600 characters each.
- "framework" for a structure with steps or sections, "hook" for an opening-line pattern, "cta" for a call-to-action pattern, "rule" for everything else.
- Keep quoted example lines in their original language; write the explanations in the article's language.
- Skip anything that only promotes the article, its author or a product."""


async def distill_article(request: ArticleDistillRequest) -> ArticleDistillResponse:
    text = (request.text or "").strip()
    if len(text) < 200:
        raise ValueError("The page did not contain enough readable text")
    client = _openai()
    user = f"URL: {request.url}\n" + (f"Field of interest: {request.field}\n" if request.field else "") + "\nARTICLE:\n" + text[:24000]
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": DISTILL_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(completion.choices[0].message.content or "{}")
    principles: list[DistilledPrinciple] = []
    for p in data.get("principles") or []:
        if not isinstance(p, dict):
            continue
        content = str(p.get("content") or "").strip()
        if len(content) < 20:
            continue
        principles.append(
            DistilledPrinciple(
                type=_pick(p.get("type"), PRINCIPLE_TYPES) or "rule",
                title=str(p.get("title") or content[:60]).strip()[:80],
                content=content[:700],
                format=_pick(p.get("format"), FORMATS),
                goal=_pick(p.get("goal"), GOALS),
            )
        )
    return ArticleDistillResponse(
        title=str(data.get("title") or "").strip()[:120],
        language=_pick(data.get("language"), LANGUAGES) or "en",
        principles=principles[:10],
    )
