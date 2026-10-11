"""
Editor pass: every draft is checked and corrected before the user sees it, against
the Derja rules of the library, a lexicon of words that are not Tunisian, the
script framework and the caption rules. The idea and the length stay; only the
craft is fixed. Used by the Studio (write_draft) and by the batch draft generator.
"""
import asyncio
import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

CRITIC_MODEL = os.getenv("STUDIO_CRITIC_MODEL", "gpt-4o")
TIMEOUT_S = 75

LEXICON = """الهاتف -> التليفون · السيارة -> الكرهبة · الآن -> توا · جداً / كثيراً -> برشا · ماذا -> شنوة · كيف -> كيفاش ·
لماذا -> علاش · أين -> وين · متى -> وقتاش · جيد -> باهي · أريد / أرغب -> نحب · يجب / ينبغي -> لازم · أيضاً -> زادة ·
بسرعة -> فيسع · دائماً -> ديما · فقط -> كهو / برك · لا تنسَ -> ما تنساش · للمزيد -> لزيد · أصدقائك -> صحابك ·
انضم إلى -> ادخل مع · السقوط -> الطيحة · يسقط -> يطيح · شاهد -> شوف · اضغط / انقر -> كليكي · اطلب -> كماندي ·
الطلب -> الكوماند · التوصيل -> livraison · الدفع عند الاستلام -> تخلّص كان كي يوصلك (COD) · هذا هو -> هاو / هذاكا ·
إزاي -> كيفاش · ليه -> علاش · كده -> هكا · دلوقتي -> توا · عايز -> نحب · شو -> شنوة · كتير -> برشا · هلق -> توا ·
بزاف -> برشا · واش / هل -> ياخي · بدي -> نحب · منيح -> باهي"""

SYSTEM = """You are the editor of Navix Studio: a native Tunisian copywriter who checks every draft before the user sees it. You receive a draft written for a Tunisian brand, the house rules and the brief. Correct it; keep its idea, its structure and its length (never more than 20% longer). Answer with JSON only: the same keys as the draft you receive (caption_text, video_script, hook_alternatives, cta_alternatives, research when present) with their corrected values, plus "changes": a list of short English notes on what you changed (empty when nothing).

Checks, in this order:
1. Language. Content language: {content_language}. Every Arabic word must be Tunisian Derja as Tunisians write it on Instagram. Replace standard-Arabic, Egyptian, Levantine or Moroccan words (use the lexicon and your own judgement). Keep the French and English words Tunisians say in Latin letters (commande, lien, livraison, promo, COD, reel, story). Keep brand and product names exactly as given. Never glue an Arabic suffix to a Latin word: write "التليفون متاعك", never "التليفونك".
2. Script (video formats only). Spoken lines only, one shot per line, in the content language, in exactly this shape: "the spoken words [camera note]". The note comes AFTER the words, never before; never two shots on one line; never a line that is only a note. For a sales script the order is: hook; the pain in everyday moments; why what the buyer uses today fails; "هذاكا علاش" with the product's mechanism (what makes it work); a proof the camera can show; one objection answered ("تقول ...؟"); one CTA that states the action, the reassurance (COD) and a believable reason for urgency. Add a missing part in one or two lines; leave the parts that are there.
3. Caption. Always return caption_text. A real Instagram caption in the content language: the first line is the hook, then two to four short lines that carry value on their own (not a summary of the video), one CTA, then 3 to 5 hashtags about THIS piece (the product, the problem, the city) and not about the account's general niche. Fix garbled phrases. When the caption is missing, empty or under 40 characters, write it from the script and the brief.
4. Hooks and CTAs. 3 to 5 each, in Derja, different patterns, no two alike.
5. No placeholders anywhere: replace [PRODUCT], [اسم المنتج], (حط الاسم) and the like with the product's name from the brief, or its generic Derja name when there is no brand name (الحامل المغناطيسي, التابي, الكريم). Never invent numbers, prices or claims that are not in the draft or the brief.
6. Relevance. Every line must be about THIS product, topic and audience. A line that would fit another product (a car mat, an olive press, skincare) was copied from an example: rewrite it for this brief or delete it. Hooks must name this product's situation.
7. Be strict on language: a Tunisian reading the result must not feel a single formal-Arabic phrase (يتطلب، الاستخدام، الرؤية، اصنع، سوف، لقد، حيث). When in doubt, rewrite the line the way a Tunisian would say it out loud.

HOUSE RULES (Navix library):
{rules}

LEXICON (not Tunisian -> Tunisian):
{lexicon}"""

FIELDS = ("caption_text", "video_script", "hook_alternatives", "cta_alternatives", "research")

_client: AsyncOpenAI | None = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=TIMEOUT_S)
    return _client


def _brief(draft: dict[str, Any], brand_kit: Any, content_language: str | None, niche: str | None) -> dict[str, Any]:
    kit = brand_kit if isinstance(brand_kit, dict) else (brand_kit.model_dump() if hasattr(brand_kit, "model_dump") else {})
    kit = kit or {}
    return {
        "format": draft.get("format"),
        "idea": draft.get("idea_title") or draft.get("direction_title"),
        "content_language": content_language or "same as my posts",
        "niche": niche,
        "tone": kit.get("tone_of_voice"),
        "audience": kit.get("target_audience"),
        "objective": kit.get("objective"),
        "words_to_avoid": kit.get("vocab_exclude"),
        "constraints": kit.get("constraints"),
    }


def _as_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return items[:5] if items else fallback
    return fallback


def _as_script(value: Any, fallback: Any) -> Any:
    if isinstance(value, list):
        return "\n".join(str(v) for v in value if str(v).strip()) or fallback
    if isinstance(value, str) and value.strip():
        return value
    return fallback


_NOTE = re.compile(r"(\[[^\]]*\])")


def normalize_script(text: Any) -> Any:
    """One shot per line as 'spoken words [camera note]': notes that came first, notes on
    their own line and several shots glued on one line are put back in shape."""
    if not isinstance(text, str) or not text.strip():
        return text
    shots: list[list[str | None]] = []
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        pending: str | None = None
        words = ""
        for part in _NOTE.split(line):
            if not part or not part.strip():
                continue
            if part.startswith("[") and part.endswith("]"):
                note = part[1:-1].strip()
                if words.strip():
                    shots.append([words.strip(), note])
                    words = ""
                elif shots and shots[-1][1] is None and pending is None:
                    shots[-1][1] = note  # a note on its own line belongs to the line above
                else:
                    pending = note  # a note before its words
            else:
                if pending is not None:
                    shots.append([part.strip(), pending])
                    pending = None
                else:
                    words = f"{words} {part.strip()}".strip()
        if words.strip():
            shots.append([words.strip(), None])
        if pending is not None:
            if shots and shots[-1][1] is None:
                shots[-1][1] = pending
            else:
                shots.append(["", pending])
    out = []
    for words, note in shots:
        if words and note:
            out.append(f"{words} [{note}]")
        elif words:
            out.append(words)
        elif note:
            out.append(f"[{note}]")
    return "\n".join(out)


async def edit_draft(
    draft: dict[str, Any],
    *,
    content_language: str | None,
    brand_kit: Any,
    rules: str,
    niche: str | None = None,
) -> dict[str, Any]:
    """Return the draft with its fields corrected; the original on any failure."""
    fields = {k: draft.get(k) for k in FIELDS if draft.get(k)}
    if not fields.get("caption_text") and not fields.get("video_script"):
        return draft
    # The caption is always part of the contract: an empty one is a defect the editor must fix
    fields.setdefault("caption_text", "")
    system = SYSTEM.format(
        content_language=content_language or "same as my posts",
        rules=rules.strip() or "(none yet)",
        lexicon=LEXICON,
    )
    user = json.dumps({"brief": _brief(draft, brand_kit, content_language, niche), "draft": fields}, ensure_ascii=False)
    try:
        completion = await asyncio.wait_for(
            _openai().chat.completions.create(
                model=CRITIC_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.2,
            ),
            timeout=TIMEOUT_S,
        )
        out = json.loads(completion.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning(f"Editor pass skipped: {exc}")
        untouched = dict(draft)
        untouched["video_script"] = normalize_script(draft.get("video_script"))
        return untouched
    fixed = dict(draft)
    if isinstance(out.get("caption_text"), str) and out["caption_text"].strip():
        fixed["caption_text"] = out["caption_text"].strip()
    if draft.get("video_script"):
        fixed["video_script"] = normalize_script(_as_script(out.get("video_script"), draft.get("video_script")))
    fixed["hook_alternatives"] = _as_list(out.get("hook_alternatives"), list(draft.get("hook_alternatives") or []))
    fixed["cta_alternatives"] = _as_list(out.get("cta_alternatives"), list(draft.get("cta_alternatives") or []))
    if isinstance(out.get("research"), dict):
        fixed["research"] = out["research"]
    changes = [str(c).strip() for c in (out.get("changes") or []) if str(c).strip()][:8]
    fixed["editor_changes"] = changes
    if changes:
        logger.info("Editor pass: %d change(s): %s", len(changes), " | ".join(changes)[:300])
    return fixed
