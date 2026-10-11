"""
RAG retrieval helpers for Navix AI agents.

Wraps the Supabase `match_embeddings` RPC and the OpenAI embeddings API.
Every agent that needs grounding (analysis, directions, drafts, brand memory)
should retrieve via `retrieve_context()` before generating output.
"""
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from supabase import Client, create_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


_supabase: Client | None = None
_openai: AsyncOpenAI | None = None


def get_supabase() -> Client:
    """Lazy-init Supabase client with service-role key (bypasses RLS for RPC calls)."""
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in ai-service .env"
            )
        _supabase = create_client(url, key)
    return _supabase


def get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai


async def embed_text(text: str) -> list[float]:
    """Generate a single embedding for a query string."""
    client = get_openai()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMS,
    )
    return response.data[0].embedding


async def retrieve_context(
    project_id: str,
    query: str,
    *,
    source_types: list[str] | None = None,
    match_count: int = 8,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks for a project via cosine similarity.

    source_types: filter to e.g. ['self_post'] for user's own voice,
                  ['competitor_post'] for market signals,
                  ['draft', 'feedback'] for brand memory.
                  None means all.
    Returns chunks ordered by similarity desc.
    """
    try:
        query_embedding = await embed_text(query)
    except Exception as e:
        logger.error(f"Embedding generation failed for project {project_id}: {e}")
        return []

    try:
        sb = get_supabase()
        result = sb.rpc(
            "match_embeddings",
            {
                "query_embedding": query_embedding,
                "match_project_id": project_id,
                "match_count": match_count,
                "match_threshold": threshold,
                "filter_source_types": source_types,
            },
        ).execute()
        chunks = result.data or []
        logger.info(
            f"RAG: retrieved {len(chunks)} chunks for project {project_id} "
            f"(types={source_types or 'all'}, query='{query[:60]}...')"
        )
        return chunks
    except Exception as e:
        logger.warning(f"RAG retrieval failed for project {project_id}: {e}")
        return []


def format_chunks(chunks: list[dict[str, Any]], header: str) -> str:
    """Render retrieved chunks for inclusion in an LLM prompt."""
    if not chunks:
        return f"{header}: (none available — generate based on general best practices)"

    lines = [f"{header}:"]
    for i, c in enumerate(chunks, 1):
        sim = c.get("similarity", 0.0)
        text = (c.get("chunk_text") or "").strip().replace("\n", " ")[:300]
        lines.append(f"  [{i}] (relevance {sim:.2f}) {text}")
    return "\n".join(lines)


async def retrieve_brand_voice(project_id: str, query: str, n: int = 6) -> str:
    """Convenience: retrieve user's own posts/drafts to ground voice."""
    chunks = await retrieve_context(
        project_id,
        query,
        source_types=["self_post", "draft", "brand_kit"],
        match_count=n,
    )
    return format_chunks(chunks, "USER'S OWN BRAND VOICE (from their posts & approved drafts)")


async def retrieve_market_context(project_id: str, query: str, n: int = 8) -> str:
    """Convenience: retrieve competitor content to ground market analysis."""
    chunks = await retrieve_context(
        project_id,
        query,
        source_types=["competitor_post"],
        match_count=n,
    )
    return format_chunks(chunks, "COMPETITOR CONTENT (recent posts in the user's market)")


async def retrieve_feedback_history(project_id: str, n: int = 10) -> str:
    """Convenience: retrieve recent feedback for brand memory updates."""
    chunks = await retrieve_context(
        project_id,
        "user preferences and feedback on content drafts",
        source_types=["feedback", "draft"],
        match_count=n,
    )
    return format_chunks(chunks, "RECENT USER FEEDBACK ON DRAFTS")


# ───────────────────────── Navix knowledge library ─────────────────────────
# Curated hooks, CTAs, frameworks, example scripts, dialect rules and distilled
# articles, shared by every project (tables knowledge_items / knowledge_chunks).

LIBRARY_HEADERS = {
    "rule": "RULES (always follow them):",
    "framework": "FRAMEWORKS (use as the structure of the script; fill every section for THIS product and audience):",
    "hook": "HOOK TEMPLATES (fill the [BRACKETS] from the brief, adapt gender and number in Derja, offer 3 to 5 filled variants):",
    "cta": "CTA PATTERNS (pick by goal and funnel stage, adapt the product words):",
    "script": "EXAMPLE SCRIPTS (match the register, rhythm and language mix; never copy the content):",
    "caption": "EXAMPLE CAPTIONS (match the register and language mix; never copy the content):",
    "article": "PRINCIPLES from articles (apply where relevant):",
}
LONG_TYPES = {"framework", "script"}


async def match_knowledge(
    embedding: list[float],
    *,
    types: list[str] | None = None,
    languages: list[str] | None = None,
    scripts: list[str] | None = None,
    formats: list[str] | None = None,
    goals: list[str] | None = None,
    field: str | None = None,
    match_count: int = 6,
    threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Nearest library items for an already computed query embedding."""
    try:
        sb = get_supabase()
        result = sb.rpc(
            "match_knowledge",
            {
                "query_embedding": embedding,
                "match_count": match_count,
                "match_threshold": threshold,
                "filter_types": types,
                "filter_languages": languages,
                "filter_scripts": scripts,
                "filter_formats": formats,
                "filter_goals": goals,
                "filter_field": field,
            },
        ).execute()
        return result.data or []
    except Exception as e:
        logger.warning(f"Knowledge retrieval failed: {e}")
        return []


async def retrieve_knowledge(query: str, **kwargs: Any) -> list[dict[str, Any]]:
    try:
        embedding = await embed_text(query)
    except Exception as e:
        logger.error(f"Embedding generation failed for a library query: {e}")
        return []
    return await match_knowledge(embedding, **kwargs)


def languages_for(content_language: str | None) -> list[str] | None:
    """Map the user's content-language choice to library languages (None = all)."""
    s = (content_language or "").lower()
    if any(k in s for k in ("derja", "tunis", "arab", "darija")):
        return ["derja", "mixed"]
    if any(k in s for k in ("french", "fran")):
        return ["fr", "mixed"]
    if "english" in s:
        return ["en", "mixed"]
    return None


def _render_item(row: dict[str, Any]) -> str:
    cap = 1500 if row.get("type") in LONG_TYPES else 400
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()[:cap]
    note = (row.get("note") or "").strip()
    head = f"{title}: " if title and not content.startswith(title) else ""
    tail = f" (why: {note})" if note else ""
    return f"{head}{content}{tail}"


async def build_library_block(
    query: str,
    *,
    content_language: str | None = None,
    niche: str | None = None,
    formats: list[str] | None = None,
    goals: list[str] | None = None,
) -> str:
    """One prompt block with the rules and the closest examples for this request."""
    if not query.strip():
        return ""
    try:
        embedding = await embed_text(query[:2000])
    except Exception as e:
        logger.warning(f"Library embedding failed: {e}")
        return ""
    langs = languages_for(content_language)
    field = (niche or "").strip().lower() or None
    rules = await match_knowledge(embedding, types=["rule"], languages=langs, field=field, match_count=8, threshold=0.0)
    frameworks = await match_knowledge(embedding, types=["framework"], field=field, formats=formats, goals=goals, match_count=2, threshold=0.12)
    examples = await match_knowledge(embedding, types=["script", "caption"], languages=langs, field=field, formats=formats, goals=goals, match_count=4, threshold=0.12)
    hooks = await match_knowledge(embedding, types=["hook"], languages=langs, field=field, goals=goals, match_count=6, threshold=0.12)
    ctas = await match_knowledge(embedding, types=["cta"], languages=langs, field=field, goals=goals, match_count=5, threshold=0.12)
    articles = await match_knowledge(embedding, types=["article"], field=field, match_count=3, threshold=0.2)

    groups = [
        ("rule", rules),
        ("framework", frameworks),
        ("hook", hooks),
        ("cta", ctas),
        ("script", [r for r in examples if r.get("type") == "script"]),
        ("caption", [r for r in examples if r.get("type") == "caption"]),
        ("article", articles),
    ]
    sections = []
    for key, rows in groups:
        if rows:
            sections.append(LIBRARY_HEADERS[key] + "\n" + "\n".join(f"- {_render_item(r)}" for r in rows))
    if not sections:
        return ""
    logger.info(
        "Library block: %s",
        ", ".join(f"{key}={len(rows)}" for key, rows in groups if rows),
    )
    return (
        "NAVIX LIBRARY (curated craft shared by every project: use it, never quote it to the user, never copy an example):\n"
        + "\n\n".join(sections)
    )


async def fetch_rules(limit: int = 25) -> str:
    """Every active rule of the library, for the prompts that must always carry them."""
    try:
        sb = get_supabase()
        result = (
            sb.table("knowledge_items")
            .select("title, content")
            .eq("type", "rule")
            .eq("status", "active")
            .order("created_at")
            .limit(limit)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        logger.warning(f"Rules fetch failed: {e}")
        rows = []
    lines = [f"- {(r.get('content') or '').strip()[:600]}" for r in rows if (r.get("content") or "").strip()]
    return "\n".join(lines) if lines else "(no rules in the library yet)"
