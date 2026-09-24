"""
Shared grounding for every agent: who the account is (persona chosen at signup)
and what its numbers say (account model computed by the backend from scraped
posts). Appended to system prompts so analysis, directions and drafts all reason
from the same facts instead of generic social-media advice.
"""
from typing import Any

PERSONA_GUIDANCE = {
    "creator": (
        "The account is a personal brand. Analyse and write as the creator speaking in the "
        "first person ('I') to their own audience; authority comes from lived experience."
    ),
    "ecommerce": (
        "The account is an online store. Content is product-led: benefits, proof, offers and "
        "objections; the brand speaks as 'we'."
    ),
    "agency": (
        "The account is an agency's client. Everything is about the CLIENT's brand and audience — "
        "never mention the agency."
    ),
}


def persona_block(persona: str | None) -> str:
    guidance = PERSONA_GUIDANCE.get((persona or "").lower())
    if not guidance:
        return ""
    return f"\n\nACCOUNT TYPE: {persona}. {guidance}"


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "?"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    """Rates arrive either as fractions (0.034) or percentages (3.4); render both as %."""
    if value is None:
        return "?"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number <= 1.0:
        number *= 100
    return f"{number:.1f}%"


def _ratio_delta(value: Any) -> str:
    """Comparisons are ratios own/field: 1.25 → +25% (1.25x), 0.6 → -40% (0.6x)."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "?"
    return f"{(ratio - 1) * 100:+.0f}% ({ratio:.2f}x)"


def account_model_block(model: dict[str, Any] | None) -> str:
    if not model:
        return ""
    own = model.get("self") or {}
    field = model.get("field") or {}
    comparison = model.get("comparison") or {}
    generated = model.get("generated_from") or {}

    lines = ["\n\nACCOUNT MODEL (computed from scraped posts — treat these numbers as ground truth):"]

    if own:
        lines.append(
            f"- Own account @{own.get('handle')}: {own.get('posts')} posts analysed; "
            f"avg {_num(own.get('avg_likes'))} likes, {_num(own.get('avg_comments'))} comments, "
            f"{_num(own.get('avg_views'), 0)} views; engagement rate {_pct(own.get('engagement_rate'))}; "
            f"{_num(own.get('posts_per_week'))} posts/week; dominant format {own.get('dominant_format') or 'unknown'}"
        )
        mix = own.get("format_mix") or []
        if mix:
            lines.append("  Own format mix: " + ", ".join(f"{m.get('format')} x{m.get('count')}" for m in mix[:5]))
        for post in (own.get("top_posts") or [])[:3]:
            caption = (post.get("caption") or "").replace("\n", " ")[:120]
            lines.append(
                f"  Own top post ({post.get('likes_count')} likes, {post.get('content_type') or 'post'}): \"{caption}\""
            )

    if field:
        lines.append(
            f"- Field ({generated.get('competitors_with_content', '?')} competitors, "
            f"{generated.get('posts', '?')} posts): engagement rate {_pct(field.get('avg_engagement_rate'))}; "
            f"{_num(field.get('posts_per_week'))} posts/week"
        )
        mix = field.get("format_mix") or []
        if mix:
            lines.append(
                "  Field format mix: "
                + ", ".join(f"{m.get('format')} {_pct(m.get('share'))}" for m in mix[:5])
            )
        tags = field.get("top_hashtags") or []
        if tags:
            lines.append("  Field hashtags: " + ", ".join(f"#{t.get('tag')}" for t in tags[:10]))
        days = field.get("posting_days") or []
        if days:
            best = max(days, key=lambda d: d.get("count") or 0)
            lines.append(f"  Field posts most on {best.get('day')}")

    if comparison:
        engagement = comparison.get("engagement_vs_field")
        cadence = comparison.get("cadence_vs_field")
        if engagement is not None:
            lines.append(f"- Own engagement vs field: {_ratio_delta(engagement)}")
        if cadence is not None:
            lines.append(f"- Own posting cadence vs field: {_ratio_delta(cadence)}")

    lines.append("Judge formats, cadence and gaps against these numbers; never contradict them.")
    return "\n".join(lines)


def creator_memory_block(memory: str | None) -> str:
    """What Navix has learned about this creator across sessions (already formatted by the backend)."""
    if not memory or not memory.strip():
        return ""
    header = "CREATOR MEMORY (learned across sessions; the user can edit it — follow it unless they say otherwise):"
    return f"\n\n{header}\n{memory.strip()}"


def grounding(persona: str | None, account_model: dict[str, Any] | None, creator_memory: str | None = None) -> str:
    """Everything an agent should know about the account before it reasons."""
    return persona_block(persona) + account_model_block(account_model) + creator_memory_block(creator_memory)
