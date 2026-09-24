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
MAX_TOOL_ROUNDS = 6
HISTORY_LIMIT = 24
# Messages longer than this are almost always specific; the router is skipped for them
ROUTER_MAX_CHARS = 220

_client: AsyncOpenAI | None = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        # A stalled stream should fail in seconds, not minutes: the UI is waiting on it
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=1)
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
            "name": "ask_user",
            "description": "Ask the user ONE decision as a small form they click (format, topic, angle, length, tone, language, which idea) instead of asking in prose. Use it only when the answer changes what you would make. After calling it, stop and wait for the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "one short question"},
                    "kind": {"type": "string", "enum": ["single", "multi", "text"]},
                    "options": {
                        "type": "array",
                        "description": "2-5 options for single/multi; omit for text",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "short label, max 6 words"},
                                "hint": {"type": "string", "description": "optional one-line hint"},
                            },
                            "required": ["label"],
                        },
                    },
                    "allow_other": {"type": "boolean", "description": "let the user type their own answer (default true)"},
                },
                "required": ["question", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_next",
            "description": "Offer 2-4 short next steps the user can click, e.g. 'Make it shorter', 'Try it as a carousel', 'Same idea in French'. Call it at the END of a substantive reply, never on its own.",
            "parameters": {
                "type": "object",
                "properties": {
                    "suggestions": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}}
                },
                "required": ["suggestions"],
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
- Ideas go through propose_ideas (cards). Drafts and revisions go through write_draft (cards). Keep chat prose short: what you did, what you noticed, one next step. Never paste full captions or scripts in prose.
- Write for a chat window: short paragraphs, **bold** the key phrase, bullets for lists, headings no bigger than ###, no walls of text.
- FIRST decide whether you know WHAT to make. If the user named neither a topic nor a format ("I want to make something this week", "help me post", "any ideas?"), your first and only action is ask_user: one short line of context, then 3-5 concrete options built from their data (their strongest format, a pillar that is working in their field, a recent post worth following up). Do not search or propose ideas before they answer. Once the topic or format is known, generate.
- Later, call ask_user again only at a real fork that changes what you would make (which idea to draft, length, language, tone). One question per turn, 2-5 clickable options, then stop and wait. If the user asks for ideas AND a draft in one message, deliver both in the same turn without asking.
- Never list choices as a numbered list in prose. Choices are always an ask_user form the user can click.
- End every substantive reply (ideas, a draft, an analysis, an answer) with suggest_next: 2-4 short next steps the user can click. Skip it only when you just asked a question.
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

    if name == "ask_user":
        kind = args.get("kind") if args.get("kind") in ("single", "multi", "text") else "single"
        options = [o for o in (args.get("options") or []) if isinstance(o, dict) and o.get("label")][:5]
        payload = {
            "question": str(args.get("question") or "").strip()[:300],
            "kind": kind,
            "options": [{"label": str(o["label"]).strip()[:80], "hint": str(o.get("hint") or "").strip()[:120]} for o in options],
            "allow_other": bool(args.get("allow_other", True)),
        }
        return (
            [{"type": "question", "question": payload}],
            "Question shown to the user as a form. Stop here and wait for their answer — do not continue generating.",
        )

    if name == "suggest_next":
        suggestions = [str(s).strip()[:60] for s in (args.get("suggestions") or []) if str(s).strip()][:4]
        return [{"type": "suggestions", "suggestions": suggestions}], "Suggestions shown as chips under your reply."

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
            "otherwise write ONE or two sentences on what the ideas have in common and which you would start with. "
            "Do not repeat the titles or list the ideas in prose — the cards already show them.",
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


SUGGEST_TOOL = next(t for t in TOOLS if t["function"]["name"] == "suggest_next")
SUGGEST_MODEL = os.getenv("STUDIO_SUGGEST_MODEL", "gpt-4o-mini")


async def _forced_suggestions(client: AsyncOpenAI, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
    """One forced suggest_next call so every substantive reply ends with clickable next steps."""
    try:
        completion = await client.chat.completions.create(
            model=SUGGEST_MODEL,
            messages=messages
            + [
                {
                    "role": "system",
                    "content": "Call suggest_next with 2-4 short next steps (max 6 words each) the user would click "
                    "right now, given what was just shown. Imperative voice, e.g. 'Draft idea 2 as a reel'.",
                }
            ],
            tools=[SUGGEST_TOOL],
            tool_choice={"type": "function", "function": {"name": "suggest_next"}},
            temperature=0.4,
        )
        call = (completion.choices[0].message.tool_calls or [None])[0]
        if call is None:
            return
        args = json.loads(call.function.arguments or "{}")
        events, _ = await _run_tool("", "suggest_next", args)
        for event in events:
            if event.get("suggestions"):
                yield _sse(event)
    except Exception as exc:  # chips are a nicety, never a failure
        logger.warning(f"Forced suggest_next failed: {exc}")


ROUTER_PROMPT = (
    "You route a content-creation chat between a creator and Navix Studio. Classify what the latest user message "
    "needs. Answer with JSON: {\"intent\": \"ask\" | \"ideas\" | \"draft\" | \"chat\"}.\n"
    "- ask: the user wants content but names neither a topic nor a format, and nothing earlier settles it "
    "(\"I want to make something this week\", \"help me post\", \"any ideas?\").\n"
    "- ideas: they want ideas, angles, suggestions or a plan, and a topic or a format is known (including when "
    "they just answered Studio's question with a format or topic).\n"
    "- draft: they want something written: a caption, script, hook, post, draft, revision, translation or a "
    "rewrite. If they want ideas AND a draft, answer draft.\n"
    "- chat: questions about their data, numbers, posts, references or Navix, feedback, thanks, small talk, "
    "anything else."
)
INTENTS = ("ask", "ideas", "draft", "chat")
# Tool calls are required for at most this many rounds while the intended card is missing
REQUIRED_ROUNDS = 4


async def _route(client: AsyncOpenAI, history: list[dict[str, Any]]) -> str:
    """Cheap router so the shape of the reply (form, idea cards, draft card) does not depend on the model's mood."""
    last = history[-1] if history else None
    if not last or last.get("role") != "user":
        return "chat"
    try:
        completion = await client.chat.completions.create(
            model=SUGGEST_MODEL,
            messages=[{"role": "system", "content": ROUTER_PROMPT}, *history[-6:]],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=12,
        )
        verdict = json.loads(completion.choices[0].message.content or "{}")
        intent = str(verdict.get("intent") or "chat")
    except Exception as exc:
        logger.warning(f"Studio router failed: {exc}")
        return "chat"
    if intent not in INTENTS:
        return "chat"
    # A long message is specific by nature: never answer it with a form
    if intent == "ask" and len(str(last.get("content") or "")) > ROUTER_MAX_CHARS:
        return "ideas"
    return intent


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
    waiting_for_user = False
    suggested = False  # suggest_next already shown this turn
    substantive = False  # the turn produced prose or cards worth following up
    try:
        # Open requests ("make something this week") start with a form, not a guess; idea and draft
        # requests must end in cards, so tool calls stay mandatory until the intended card is on screen.
        intent = await _route(client, messages[1:])
        delivered: set[str] = set()  # card types emitted so far this turn
        for round_index in range(MAX_TOOL_ROUNDS):
            tool_choice: Any = "auto"
            if round_index == 0 and intent == "ask":
                tool_choice = {"type": "function", "function": {"name": "ask_user"}}
            elif (
                intent in ("ideas", "draft")
                and round_index < REQUIRED_ROUNDS
                and not suggested
                and ("ideas" if intent == "ideas" else "draft") not in delivered
            ):
                tool_choice = "required"
            stream = await client.chat.completions.create(
                model=STUDIO_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
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

            if text.strip():
                substantive = True

            if not calls:
                if text:
                    messages.append({"role": "assistant", "content": text})
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
                    if event.get("type") in ("ideas", "draft"):
                        substantive = True
                        delivered.add(str(event["type"]))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                if c["name"] == "ask_user":
                    waiting_for_user = True
                elif c["name"] == "suggest_next":
                    suggested = True
            # A form is on screen: the turn ends here and the user's answer opens the next one
            if waiting_for_user:
                break

        # The model tends to forget the closing chips; ask for them explicitly with a cheap call.
        if substantive and not suggested and not waiting_for_user:
            async for line in _forced_suggestions(client, messages):
                yield line
        yield _sse({"type": "done"})
    except Exception as exc:
        logger.error(f"Studio turn failed for project {request.project_id}: {exc}")
        yield _sse({"type": "error", "message": str(exc)})
