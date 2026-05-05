import json
import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from schemas.brand_memory import BrandMemoryRequest, BrandMemoryResponse

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=2048)


class MemoryState(TypedDict):
    request: BrandMemoryRequest
    raw_analysis: str
    final_result: BrandMemoryResponse | None


SYSTEM_PROMPT = """You are a brand learning engine. You analyze user feedback on content drafts to identify patterns in what the brand owner likes and dislikes.

Your job is to extract actionable preferences that will improve future content generation:
- From APPROVED drafts: what tone, style, hooks, and CTAs does the user prefer?
- From REJECTED drafts: what should be avoided?
- From EDITED drafts: what did the user change? What does that tell us about their preferences?

Be specific and actionable. Don't repeat what's already in the brand kit — only suggest NEW learnings."""


async def analyze_feedback(state: MemoryState) -> MemoryState:
    """Analyze feedback patterns to extract brand preferences."""
    request = state["request"]
    current_kit = request.current_brand_kit

    current_kit_summary = f"""CURRENT BRAND KIT:
- Tone: {current_kit.tone_of_voice or 'Not set'}
- Formality: {current_kit.formality_level or 'Not set'}
- Audience: {current_kit.target_audience or 'Not set'}
- Objective: {current_kit.objective or 'Not set'}
- Excluded words: {current_kit.vocab_exclude or 'None'}
- Constraints: {', '.join(current_kit.constraints) if current_kit.constraints else 'None'}
- Preferred CTA: {current_kit.preferred_cta or 'Not set'}"""

    feedback_summary = "\n\n".join(
        f"Draft: \"{(f.caption_text or 'no text')[:200]}\"\nAction: {f.action}\nNotes: {f.notes or 'none'}"
        for f in request.feedback
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""{current_kit_summary}

USER FEEDBACK ON RECENT DRAFTS:
{feedback_summary}

Based on this feedback, identify:
1. What brand kit fields should be updated (only fields that need changing)
2. What NEW patterns were learned (things not already in the brand kit)

Return ONLY JSON:
{{
  "updated_preferences": {{
    "tone_of_voice": "...",  // only include if should change
    "vocab_exclude": "...",  // append new words to avoid
    "preferred_cta": "..."   // only if pattern is clear
  }},
  "learned_patterns": [
    "User prefers ...",
    "User dislikes ...",
    "..."
  ]
}}

Only include fields in updated_preferences that actually need changing. Leave out fields that are fine as-is.""")
    ])

    state["raw_analysis"] = response.content
    return state


async def parse_memory_update(state: MemoryState) -> MemoryState:
    """Parse the memory update into structured response."""
    raw = state["raw_analysis"].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    parsed = json.loads(raw)
    state["final_result"] = BrandMemoryResponse(**parsed)
    return state


def build_memory_graph():
    graph = StateGraph(MemoryState)

    graph.add_node("analyze_feedback", analyze_feedback)
    graph.add_node("parse_memory_update", parse_memory_update)

    graph.set_entry_point("analyze_feedback")
    graph.add_edge("analyze_feedback", "parse_memory_update")
    graph.add_edge("parse_memory_update", END)

    return graph.compile()


memory_graph = build_memory_graph()


async def run_brand_memory_update(request: BrandMemoryRequest) -> BrandMemoryResponse:
    """Run the brand memory update pipeline."""
    logger.info(f"Updating brand memory for project {request.project_id} with {len(request.feedback)} feedback items")

    initial_state: MemoryState = {
        "request": request,
        "raw_analysis": "",
        "final_result": None,
    }

    result = await memory_graph.ainvoke(initial_state)

    if result["final_result"] is None:
        raise ValueError("Brand memory update failed to produce a result")

    logger.info(f"Brand memory updated for project {request.project_id}: {len(result['final_result'].learned_patterns)} patterns learned")
    return result["final_result"]
