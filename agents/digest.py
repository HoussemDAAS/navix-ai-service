"""
Weekly digest: one honest headline and one grounded idea for the Home page,
written from facts the backend computed (follower and engagement deltas, the
creator's best post, the breakout post in their field, the pipeline).
"""
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

from prompt_context import account_model_block, creator_memory_block
from schemas.digest import DigestIdea, WeeklyDigestRequest, WeeklyDigestResponse

logger = logging.getLogger(__name__)

DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gpt-4o")
FORMATS = {"reel", "carousel", "image", "story", "video", "live"}

SYSTEM_PROMPT = """You write the weekly digest on the Home page of ONE creator or brand, from facts computed by Navix. You speak TO them ("you", "your"), never as them. Two things:

1. headline: one sentence, at most 140 characters, plain English, second person, with the single most important thing that happened this week and one real number from the facts (followers moved, how your best post did against your usual, how active your field was). Honest and warm: if they did not post, say so plainly without scolding. No emoji, no hype.

2. idea: the ONE piece of content worth making next, grounded in a fact: the subject of their best post (a follow-up, a second angle, the same format on a new topic), the angle of the breakout post in their field (adapted to their voice, never copied), or a gap the field left open this week. Fields:
   - title: at most 8 words, no quotes, names the concrete subject (not "personal growth stories" but the actual topic from the caption).
   - why: one sentence, second person, that cites the exact fact it comes from: the post's subject and its ×N against the usual, the handle, or the format.
   - format: one of reel, carousel, image, story, video, live, or null.
   - studio_prompt: the exact message the user would send to Navix Studio to make it. First person as the user, at most 220 characters, in English, naming the concrete topic taken from the facts, the format and the angle. Example shape: "Make me a reel that follows up my Tunisia summer post: 3 things that surprised me, same warm tone."

Rules: never invent numbers, posts or handles; only use what the facts contain. Respect the brand kit and the creator memory. Return ONLY JSON: {"headline": "...", "idea": {"title": "...", "why": "...", "format": "...", "studio_prompt": "..."}}"""


def _trim(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _format_brand_kit(kit: dict | None) -> str:
    if not kit:
        return "(not set)"
    keys = ("tone_of_voice", "formality_level", "target_audience", "objective", "preferred_cta")
    return ", ".join(f"{k}: {kit[k]}" for k in keys if kit.get(k)) or "(not set)"


async def write_weekly_digest(request: WeeklyDigestRequest) -> WeeklyDigestResponse:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=45.0, max_retries=1)
    p = request.project
    # The persona guidance is skipped on purpose: it tells content agents to write AS the creator,
    # while the digest talks TO them.
    prompt = (
        f"PROJECT: {p.name or 'unknown'} · account type: {request.persona or 'unknown'}"
        f" · niche: {p.niche or 'unknown'} · handle: @{p.handle or '?'}"
        f" · content language: {p.language or 'same as their posts'}\n"
        f"BRAND KIT: {_format_brand_kit(request.brand_kit)}"
        f"{account_model_block(request.account_model)}{creator_memory_block(request.creator_memory)}\n\n"
        f"THIS WEEK'S FACTS (JSON):\n{json.dumps(request.facts, ensure_ascii=False, default=str)}\n\n"
        "Reminder: headline and why address the creator as 'you'; only studio_prompt is in the first person."
    )
    completion = await client.chat.completions.create(
        model=DIGEST_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.5,
    )
    raw = json.loads(completion.choices[0].message.content or "{}")
    idea_raw = raw.get("idea") if isinstance(raw.get("idea"), dict) else None
    idea = None
    if idea_raw and idea_raw.get("title") and idea_raw.get("studio_prompt"):
        fmt = str(idea_raw.get("format") or "").lower().strip() or None
        idea = DigestIdea(
            title=_trim(idea_raw.get("title"), 80),
            why=_trim(idea_raw.get("why"), 240),
            format=fmt if fmt in FORMATS else None,
            studio_prompt=_trim(idea_raw.get("studio_prompt"), 260),
        )
    return WeeklyDigestResponse(headline=_trim(raw.get("headline"), 180) or None, idea=idea)
