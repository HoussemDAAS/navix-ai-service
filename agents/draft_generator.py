import json
import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from schemas.drafts import DraftsRequest, DraftsResponse, Draft
from rag import retrieve_brand_voice
from prompt_context import grounding

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0.8, max_tokens=8192)


class DraftState(TypedDict):
    request: DraftsRequest
    brand_voice_context: str
    raw_drafts: str
    final_result: DraftsResponse | None


async def fetch_brand_voice(state: DraftState) -> DraftState:
    """Retrieve user's own captions/drafts so generated content sounds like them."""
    request = state["request"]
    direction_titles = " | ".join(d.title_pillar for d in request.directions[:5])
    query = f"writing style and tone for posts about {direction_titles}"
    state["brand_voice_context"] = await retrieve_brand_voice(request.project_id, query, n=10)
    return state


SYSTEM_PROMPT = """You are an expert social media copywriter. You write engaging, scroll-stopping content that matches a brand's voice perfectly.

Your writing must:
- Match the brand's tone and formality exactly
- Never use excluded words/phrases
- Respect all brand constraints
- Include hooks that grab attention in the first line
- End with clear, compelling CTAs
- Feel natural and human — never robotic or generic
- Be platform-appropriate (Instagram captions are different from TikTok scripts)

For video scripts: write a brief shot-by-shot plan with spoken text and visual notes.
For captions: write the full caption text ready to post."""


async def generate_drafts(state: DraftState) -> DraftState:
    """Generate drafts for each content direction."""
    request = state["request"]
    brand_kit = request.brand_kit

    brand_rules = f"""BRAND VOICE RULES:
- Tone: {brand_kit.tone_of_voice or 'Casual'}
- Formality: {brand_kit.formality_level or 'Neutral'}
- Target Audience: {brand_kit.target_audience or 'General'}
- Content Objective: {brand_kit.objective or 'Engage and grow'}
- NEVER use these words: {brand_kit.vocab_exclude or 'none specified'}
- Constraints: {', '.join(brand_kit.constraints) if brand_kit.constraints else 'none'}
- Preferred CTA style: {brand_kit.preferred_cta or 'No preference'}"""

    directions_text = "\n\n".join(
        f"""DIRECTION {i+1}: "{d.title_pillar}"
- Angle: {d.angle}
- Format: {d.format}
- Suggested hook: {d.hook_suggestion}
- Suggested CTA: {d.cta_suggestion}"""
        for i, d in enumerate(request.directions)
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT + grounding(request.persona, request.account_model)),
        HumanMessage(content=f"""{brand_rules}

{state['brand_voice_context']}

Use the user's own posts above as the PRIMARY voice reference. Mirror their sentence rhythm, vocabulary choices, emoji usage, and CTA patterns. The generated content must sound like the same person wrote it.

CONTENT DIRECTIONS TO DRAFT:
{directions_text}

For EACH direction above, write:
1. A full caption/post text (ready to publish)
2. A video script if the format is video/reel (shot-by-shot with spoken text), or null if static
3. 2 alternative hooks (different opening lines)
4. 2 alternative CTAs

Return ONLY a JSON array:
[
  {{
    "direction_title": "...",
    "caption_text": "...",
    "video_script": "..." or null,
    "hook_alternatives": ["...", "..."],
    "cta_alternatives": ["...", "..."]
  }}
]

Write like a real human creator — not a corporate bot. Make it engaging and authentic.""")
    ])

    state["raw_drafts"] = response.content
    return state


async def parse_drafts(state: DraftState) -> DraftState:
    """Parse the raw LLM response into structured drafts."""
    raw = state["raw_drafts"].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    parsed = json.loads(raw)
    drafts = [Draft(**d) for d in parsed]
    state["final_result"] = DraftsResponse(drafts=drafts)
    return state


def build_draft_graph():
    graph = StateGraph(DraftState)

    graph.add_node("fetch_brand_voice", fetch_brand_voice)
    graph.add_node("generate_drafts", generate_drafts)
    graph.add_node("parse_drafts", parse_drafts)

    graph.set_entry_point("fetch_brand_voice")
    graph.add_edge("fetch_brand_voice", "generate_drafts")
    graph.add_edge("generate_drafts", "parse_drafts")
    graph.add_edge("parse_drafts", END)

    return graph.compile()


draft_graph = build_draft_graph()


async def run_draft_generation(request: DraftsRequest) -> DraftsResponse:
    """Run the draft generation pipeline."""
    logger.info(f"Generating drafts for {len(request.directions)} directions in project {request.project_id}")

    initial_state: DraftState = {
        "request": request,
        "brand_voice_context": "",
        "raw_drafts": "",
        "final_result": None,
    }

    result = await draft_graph.ainvoke(initial_state)

    if result["final_result"] is None:
        raise ValueError("Draft generation failed to produce a result")

    logger.info(f"Generated {len(result['final_result'].drafts)} drafts for project {request.project_id}")
    return result["final_result"]
