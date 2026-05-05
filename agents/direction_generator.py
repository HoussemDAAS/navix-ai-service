import json
import logging
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from schemas.directions import DirectionsRequest, DirectionsResponse, ContentDirection
from schemas.analysis import AnalysisResponse

logger = logging.getLogger(__name__)

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=4096)


class DirectionState(TypedDict):
    request: DirectionsRequest
    raw_directions: str
    final_result: DirectionsResponse | None


SYSTEM_PROMPT = """You are a creative content strategist for social media. You generate strategic content directions (pillars) based on market analysis and brand identity.

Your directions must be:
- Differentiated: leverage whitespace opportunities identified in the analysis
- Brand-aligned: respect the brand's tone, audience, and constraints
- Actionable: each direction should be specific enough to write a post from
- Diverse: cover different angles, formats, and goals
- Grounded: justified by actual market data, not generic advice

Each direction is a content "pillar" — a strategic angle the brand can repeatedly use."""


async def generate_directions(state: DirectionState) -> DirectionState:
    """Generate content directions from analysis + brand kit."""
    request = state["request"]
    analysis = request.analysis
    brand_kit = request.brand_kit

    analysis_summary = f"""MARKET ANALYSIS FINDINGS:

Dominant Formats:
{chr(10).join(f'- {f.format}: {f.frequency}, {f.avg_engagement} engagement' for f in analysis.dominant_formats)}

Winning Hooks:
{chr(10).join(f'- "{h.hook_text}" ({h.pattern}, {h.effectiveness})' for h in analysis.winning_hooks)}

Whitespace Opportunities:
{chr(10).join(f'- {o.area}: {o.reasoning} (confidence: {o.confidence})' for o in analysis.whitespace_opportunities)}

Content Cadence: {analysis.content_cadence}

Key Takeaways:
{chr(10).join(f'- {t}' for t in analysis.key_takeaways)}"""

    brand_context = f"""BRAND KIT:
- Tone: {brand_kit.tone_of_voice or 'Not specified'}
- Formality: {brand_kit.formality_level or 'Not specified'}
- Target Audience: {brand_kit.target_audience or 'Not specified'}
- Objective: {brand_kit.objective or 'Not specified'}
- Words to avoid: {brand_kit.vocab_exclude or 'None'}
- Constraints: {', '.join(brand_kit.constraints) if brand_kit.constraints else 'None'}
- Preferred CTA style: {brand_kit.preferred_cta or 'Not specified'}"""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""{analysis_summary}

{brand_context}

Generate exactly {request.num_directions} content directions. Each direction should:
1. Have a clear title/pillar name
2. Specify a unique angle (perspective or approach)
3. Recommend a specific format (reel, carousel, story, static post, etc.)
4. Suggest a hook (opening line)
5. Suggest a CTA (call-to-action)
6. Include a rationale explaining WHY this direction will work based on the analysis

Make directions diverse — cover different formats, tones, and goals. Prioritize whitespace opportunities.

Return ONLY a JSON array:
[
  {{
    "title_pillar": "...",
    "angle": "...",
    "format": "...",
    "hook_suggestion": "...",
    "cta_suggestion": "...",
    "rationale": "..."
  }}
]""")
    ])

    state["raw_directions"] = response.content
    return state


async def parse_directions(state: DirectionState) -> DirectionState:
    """Parse the raw LLM response into structured directions."""
    raw = state["raw_directions"].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    parsed = json.loads(raw)
    directions = [ContentDirection(**d) for d in parsed]
    state["final_result"] = DirectionsResponse(directions=directions)
    return state


def build_direction_graph():
    graph = StateGraph(DirectionState)

    graph.add_node("generate_directions", generate_directions)
    graph.add_node("parse_directions", parse_directions)

    graph.set_entry_point("generate_directions")
    graph.add_edge("generate_directions", "parse_directions")
    graph.add_edge("parse_directions", END)

    return graph.compile()


direction_graph = build_direction_graph()


async def run_direction_generation(request: DirectionsRequest) -> DirectionsResponse:
    """Run the direction generation pipeline."""
    logger.info(f"Generating {request.num_directions} directions for project {request.project_id}")

    initial_state: DirectionState = {
        "request": request,
        "raw_directions": "",
        "final_result": None,
    }

    result = await direction_graph.ainvoke(initial_state)

    if result["final_result"] is None:
        raise ValueError("Direction generation failed to produce a result")

    logger.info(f"Generated {len(result['final_result'].directions)} directions for project {request.project_id}")
    return result["final_result"]
