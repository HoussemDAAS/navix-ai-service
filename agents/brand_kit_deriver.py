"""
Derive a brand kit from the account's own posts — the voice every other agent
must write in. Grounded ONLY in the captions provided: the model is told to
leave a field empty rather than guess when the posts do not support it.
"""
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from schemas.brand_kit import BrandKitDeriveRequest, BrandKitDeriveResponse, BrandKitPost

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=1500)

PERSONA_GUIDANCE = {
    "creator": "This is a personal brand: the creator speaking as 'I' to their own audience.",
    "ecommerce": "This is an online store: the brand speaks as 'we' and sells products — benefits, proof, offers.",
    "agency": "This is an agency's client account: describe the CLIENT's voice as it appears in the posts, not the agency's.",
}

RESPONSE_FIELDS = (
    "tone_of_voice",
    "formality_level",
    "target_audience",
    "objective",
    "preferred_cta",
    "vocab_exclude",
    "constraints",
    "evidence",
    "confidence",
)

SYSTEM_PROMPT = """You are a brand-voice analyst. You read an account's real posts and describe how it actually speaks, so a writing assistant can match it.

Rules:
- Ground every field in the posts you are given, and quote them as evidence.
- If the posts do not support a field, return null for it — never invent.
- Detect the language mix the account really writes in (for example Tunisian Arabic + French + English) and keep it in the constraints.
- Be concrete: "short punchy lines, one idea per post, ends with a question" beats "engaging".
Return ONLY JSON."""


def _format_post(post: BrandKitPost) -> str:
    stats = f"{post.likes or 0} likes, {post.comments or 0} comments"
    if post.views:
        stats += f", {post.views} views"
    tags = f" | #{' #'.join(post.hashtags[:6])}" if post.hashtags else ""
    return f"- [{post.format or 'post'} | {stats}{tags}] \"{post.caption.strip()[:320]}\""


def _snap(value, options: list[str]) -> str | None:
    """The UI offers fixed tiles: map the model's answer onto them, case-insensitively."""
    if not value:
        return None
    wanted = str(value).strip().lower()
    for option in options:
        if option.lower() == wanted:
            return option
    return None


async def derive_brand_kit(request: BrandKitDeriveRequest) -> BrandKitDeriveResponse:
    posts = "\n".join(_format_post(p) for p in request.posts[:40])
    guidance = PERSONA_GUIDANCE.get(request.persona or "", "")
    profile = request.profile

    prompt = f"""ACCOUNT
Name: {profile.name or 'unknown'}
Bio: {profile.bio or 'none'}
Niche: {profile.niche or 'unknown'}
Account type: {request.persona or 'unknown'}. {guidance}

THEIR REAL POSTS ({len(request.posts)}):
{posts}

Derive the brand kit. Allowed values:
- tone_of_voice: one of {request.tone_options}
- formality_level: one of {request.formality_options}

JSON shape:
{{
  "tone_of_voice": "one of the allowed values",
  "formality_level": "one of the allowed values",
  "target_audience": "who these posts speak to — one sentence, inferred from the content",
  "objective": "what the account is visibly trying to achieve — one sentence",
  "preferred_cta": "the call-to-action they actually use most, verbatim if possible, else null",
  "vocab_exclude": "comma-separated words or phrases that would clearly be off-brand for this voice, else null",
  "constraints": ["3-6 rules the account visibly follows: language mix, emoji use, caption length, hashtag habits, person (I/we), recurring structure"],
  "evidence": ["3-6 short verbatim quotes from the posts that justify the read"],
  "confidence": 0.0
}}"""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(raw)

    parsed["tone_of_voice"] = _snap(parsed.get("tone_of_voice"), request.tone_options)
    parsed["formality_level"] = _snap(parsed.get("formality_level"), request.formality_options)
    result = BrandKitDeriveResponse(**{k: v for k, v in parsed.items() if k in RESPONSE_FIELDS})

    logger.info(
        f"Brand kit derived for project {request.project_id} from {len(request.posts)} posts "
        f"(confidence {result.confidence})"
    )
    return result
