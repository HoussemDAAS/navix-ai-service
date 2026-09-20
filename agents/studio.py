"""
Navix Studio — the conversational co-creation agent.

One streaming turn: the model talks to the user in English, searches the
account's own posts and the field through tools, and delivers ideas and drafts
as structured cards (tool calls) instead of prose. Events are yielded as
Server-Sent Events so the backend can persist cards as they arrive.
"""
import json
import logging
import os
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from prompt_context import grounding
from rag import get_supabase, retrieve_context
from schemas.studio import StudioChatRequest

logger = logging.getLogger(__name__)

STUDIO_MODEL = os.getenv("STUDIO_MODEL", "gpt-4o")
MAX_TOOL_ROUNDS = 4
HISTORY_LIMIT = 24

_client: AsyncOpenAI | None = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_my_posts",
            "description": "Search the user's own scraped posts for their voice, recurring topics and phrasings. Use it before writing anything in their voice.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_field",
            "description": "Search competitors' posts in the user's market for angles, hooks and formats that perform.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_reference",
            "description": "Read the full text of a reference the user shared in this session (transcript, PDF text, image description). Use it before modelling content on that reference.",
            "parameters": {
                "type": "object",
                "properties": {"reference_id": {"type": "string"}},
                "required": ["reference_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_ideas",
            "description": "Show 2-5 content ideas to the user as cards. Call this whenever you suggest ideas — never list ideas only in prose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title_pillar": {"type": "string", "description": "short idea title"},
                                "angle": {"type": "string", "description": "the specific take, one or two sentences"},
                                "format": {"type": "string", "description": "reel | carousel | image | story | tiktok | short"},
                                "hook_suggestion": {"type": "string"},
                                "cta_suggestion": {"type": "string"},
                                "rationale": {"type": "string", "description": "why it fits THIS account, citing a number or a post"},
                            },
                            "required": ["title_pillar", "angle", "format", "hook_suggestion", "cta_suggestion", "rationale"],
                        },
                    }
                },
                "required": ["ideas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_draft",
            "description": "Deliver a finished piece of content as a draft card. Call this for every draft and every revision — never paste a full caption or script only in prose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "idea_title": {"type": "string", "description": "the idea this draft develops"},
                    "format": {"type": "string", "description": "reel | carousel | image | story | tiktok | short"},
                    "language": {"type": "string", "description": "language the content is written in"},
                    "caption_text": {"type": "string", "description": "the caption, ready to post"},
                    "video_script": {
                        "type": "string",
                        "description": "for video formats: shot-by-shot script as numbered lines with spoken text and visual notes; omit otherwise",
                    },
                    "hook_alternatives": {"type": "array", "items": {"type": "string"}},
                    "cta_alternatives": {"type": "array", "items": {"type": "string"}},
                    "tips": {
                        "type": "object",
                        "properties": {
                            "why_this_format": {"type": "string"},
                            "best_time": {"type": "string"},
                            "modelled_on": {"type": "string"},
                            "watch_out": {"type": "string"},
                        },
                        "required": ["why_this_format", "best_time", "modelled_on", "watch_out"],
                    },
                    "revises_draft_id": {
                        "type": "string",
                        "description": "id of the draft this revises, when the user asked to change an existing draft",
                    },
                },
                "required": ["idea_title", "format", "language", "caption_text", "hook_alternatives", "cta_alternatives", "tips"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are Navix Studio: a content strategist and copywriter who works inside the user's own data.

How you work:
- You talk to the user in English. The CONTENT you write (captions, scripts, hooks, CTAs) is in the language they chose: {content_language}. When that is "same as my posts", write in the language mix visible in their brand kit constraints and their own posts.
- Ground everything in this account: its brand kit, its numbers, its own posts (search_my_posts) and its field (search_field). When you lean on a number or a post, say so in one short clause.
- Ideas go through propose_ideas (cards). Drafts and revisions go through write_draft (cards). Keep chat prose short: what you did, what you noticed, one question or one next step. Never paste full captions or scripts in prose.
- When a request is vague, ask ONE sharp question before generating. When it is clear, generate. If the user asks for ideas AND a draft in one message, deliver both in the same turn.
- Never invent metrics, quotes or posts. If a search found nothing useful, say so.
- Tips on a draft must be specific to THIS account: why this format here, when to post (from the field data), what it was modelled on, one thing to watch out for.
- A revision keeps what the user liked and changes only what they asked; pass revises_draft_id.
{grounding}

BRAND KIT:
{brand_kit}

CREATOR MEMORY (what Navix has learned about this creator so far):
{creator_memory}"""


def _format_brand_kit(kit: dict[str, Any] | None) -> str:
    if not kit:
        return "Not set yet — infer the voice from their posts and say you are doing so."
    constraints = kit.get("constraints") or []
    return "\n".join(
        [
            f"- Tone: {kit.get('tone_of_voice') or 'not set'}",
            f"- Formality: {kit.get('formality_level') or 'not set'}",
            f"- Audience: {kit.get('target_audience') or 'not set'}",
            f"- Objective: {kit.get('objective') or 'not set'}",
            f"- Preferred CTA: {kit.get('preferred_cta') or 'not set'}",
            f"- Words to avoid: {kit.get('vocab_exclude') or 'none'}",
            f"- Constraints: {', '.join(constraints) if constraints else 'none'}",
        ]
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _format_hits(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No matching posts found."
    lines = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("chunk_text") or "").strip().replace("\n", " ")[:400]
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


async def _run_tool(project_id: str, name: str, args: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Execute a tool. Returns (events to stream to the client, text result for the model)."""
    # Own captions are short and multilingual: cosine scores against a query sit around
    # 0.3-0.4, so the default 0.4 cut-off would hide the user's own voice.
    if name == "search_my_posts":
        query = str(args.get("query") or "")
        chunks = await retrieve_context(project_id, query, source_types=["self_post"], match_count=6, threshold=0.25)
        return [{"type": "tool", "name": name, "query": query, "hits": len(chunks)}], _format_hits(chunks)

    if name == "search_field":
        query = str(args.get("query") or "")
        chunks = await retrieve_context(project_id, query, source_types=["competitor_post"], match_count=8, threshold=0.3)
        return [{"type": "tool", "name": name, "query": query, "hits": len(chunks)}], _format_hits(chunks)

    if name == "read_reference":
        reference_id = str(args.get("reference_id") or "")
        try:
            result = (
                get_supabase()
                .table("references")
                .select("title, kind, summary, extracted_text")
                .eq("id", reference_id)
                .eq("project_id", project_id)
                .maybe_single()
                .execute()
            )
            row = result.data if result is not None else None
        except Exception as exc:
            logger.warning(f"read_reference failed: {exc}")
            row = None
        if not row:
            return [{"type": "tool", "name": name, "query": reference_id, "hits": 0}], "Reference not found."
        body = (row.get("extracted_text") or row.get("summary") or "")[:6000]
        return (
            [{"type": "tool", "name": name, "query": str(row.get("title") or reference_id), "hits": 1}],
            f"REFERENCE \"{row.get('title')}\" ({row.get('kind')}):\n{body}",
        )

    if name == "propose_ideas":
        ideas = [i for i in (args.get("ideas") or []) if isinstance(i, dict)][:5]
        titles = ", ".join(str(i.get("title_pillar")) for i in ideas)
        return (
            [{"type": "ideas", "ideas": ideas}],
            f"Ideas shown to the user as cards: {titles}. If the user also asked for a draft, call write_draft now for the one they meant; "
            "otherwise say in one or two sentences what the ideas have in common and ask which one to develop.",
        )

    if name == "write_draft":
        draft = dict(args)
        draft["hook_alternatives"] = [str(h) for h in (draft.get("hook_alternatives") or [])][:5]
        draft["cta_alternatives"] = [str(c) for c in (draft.get("cta_alternatives") or [])][:5]
        return (
            [{"type": "draft", "draft": draft}],
            "Draft shown to the user as a card. In one or two sentences say what you did and offer one concrete next step (a revision angle, a second format, or approval).",
        )

    return [], f"Unknown tool {name}"


async def stream_studio_turn(request: StudioChatRequest) -> AsyncIterator[str]:
    """Yield SSE lines for one assistant turn."""
    system = SYSTEM_PROMPT.format(
        content_language=request.content_language or "same as my posts",
        grounding=grounding(request.persona, request.account_model),
        brand_kit=_format_brand_kit(request.brand_kit),
        creator_memory=request.creator_memory or "Nothing yet — this is one of the first sessions.",
    )
    if request.project:
        p = request.project
        system += (
            f"\n\nPROJECT: {p.get('name') or 'unknown'} · niche: {p.get('niche') or 'unknown'}"
            f" · handle: @{p.get('handle') or '?'} · location: {p.get('location') or 'unknown'}"
        )
    if request.references:
        listed = "\n".join(
            f"- [{r.get('id')}] {r.get('kind')} · {r.get('title')} — {(r.get('summary') or 'no summary')[:300]}"
            for r in request.references
        )
        system += (
            "\n\nREFERENCES the user shared in this session (call read_reference with the id for the full text "
            "before modelling anything on one of them):\n" + listed
        )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in request.messages[-HISTORY_LIMIT:]:
        if m.role in ("user", "assistant") and m.content:
            messages.append({"role": m.role, "content": m.content})

    client = _openai()
    try:
        for _round in range(MAX_TOOL_ROUNDS):
            stream = await client.chat.completions.create(
                model=STUDIO_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                stream=True,
            )
            text = ""
            calls: dict[int, dict[str, str]] = {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    text += delta.content
                    yield _sse({"type": "token", "text": delta.content})
                for tc in delta.tool_calls or []:
                    entry = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

            if not calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                        }
                        for c in calls.values()
                    ],
                }
            )
            for c in calls.values():
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    events, result = await _run_tool(request.project_id, c["name"], args)
                except Exception as exc:  # a failed tool must not kill the turn
                    logger.warning(f"Studio tool {c['name']} failed: {exc}")
                    events, result = [], f"Tool {c['name']} failed: {exc}"
                for event in events:
                    yield _sse(event)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        yield _sse({"type": "done"})
    except Exception as exc:
        logger.error(f"Studio turn failed for project {request.project_id}: {exc}")
        yield _sse({"type": "error", "message": str(exc)})
