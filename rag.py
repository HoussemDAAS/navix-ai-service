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
