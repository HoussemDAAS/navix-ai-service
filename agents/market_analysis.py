import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from schemas.analysis import AnalysisRequest, AnalysisResponse, FormatInsight, HookInsight, Opportunity
from rag import retrieve_brand_voice, retrieve_market_context
from prompt_context import grounding

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=4096)


class AnalysisState(TypedDict):
    request: AnalysisRequest
    market_context: str
    brand_voice_context: str
    formats_raw: str
    hooks_raw: str
    opportunities_raw: str
    final_result: AnalysisResponse | None


async def gather_rag_context(state: AnalysisState) -> AnalysisState:
    """Pull retrieved context once, reuse across all analysis steps."""
    request = state["request"]
    niche = request.brand_kit.target_audience or request.brand_kit.objective or "social media content"
    state["market_context"] = await retrieve_market_context(
        request.project_id, f"top performing posts in {niche}"
    )
    state["brand_voice_context"] = await retrieve_brand_voice(
        request.project_id, f"brand voice for {niche}"
    )
    return state


SYSTEM_PROMPT = """You are a social media market intelligence analyst. You analyze competitor content to identify patterns, winning strategies, and whitespace opportunities.

Your analysis must be:
- Data-driven: base insights on the actual signals (posts) provided
- Actionable: each insight should suggest what the brand could do differently
- Specific: avoid generic advice, reference actual patterns you observe
- Structured: follow the exact output format requested

When analyzing, consider: posting cadence, content formats (reels, carousels, stories, static), hook patterns, CTA patterns, engagement levels, and content themes."""


async def analyze_formats(state: AnalysisState) -> AnalysisState:
    """Analyze dominant content formats from competitor signals."""
    request = state["request"]

    competitors_summary = "\n".join(
        f"- @{c.handle} ({c.platform}): {c.followers_count or '?'} followers, {c.biography or 'no bio'}"
        for c in request.competitors
    )

    signals_summary = "\n".join(
        f"- @{s.competitor_handle}: \"{(s.caption or '')[:100]}...\" | {s.likes_count or 0} likes, {s.views_count or 0} views"
        for s in request.signals[:50]  # Limit to 50 signals for context window
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT + grounding(request.persona, request.account_model)),
        HumanMessage(content=f"""Analyze the dominant content formats from these competitors and their posts.

COMPETITORS:
{competitors_summary}

RECENT POSTS (signals):
{signals_summary}

{state['market_context']}

Identify the top 3-5 content formats being used. For each format, note:
1. The format name (e.g., "Short-form educational reels", "Carousel breakdowns", "Story polls")
2. How frequently it appears (e.g., "40% of posts", "2-3 per week")
3. Average engagement level (e.g., "High - 3x average likes", "Medium")
4. 1-2 specific examples from the posts above

Also determine the overall content cadence (how often competitors post on average).

Return your analysis as a structured text. Be specific and reference actual data.""")
    ])

    state["formats_raw"] = response.content
    return state


async def analyze_hooks_ctas(state: AnalysisState) -> AnalysisState:
    """Analyze winning hooks and CTAs from signals."""
    request = state["request"]

    high_engagement_signals = sorted(
        [s for s in request.signals if s.likes_count],
        key=lambda s: s.likes_count or 0,
        reverse=True
    )[:30]

    captions = "\n".join(
        f"[{s.likes_count} likes] @{s.competitor_handle}: \"{(s.caption or '')[:200]}\""
        for s in high_engagement_signals
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT + grounding(request.persona, request.account_model)),
        HumanMessage(content=f"""Analyze the hooks (opening lines) and CTAs (calls-to-action) from these top-performing posts.

TOP PERFORMING POSTS (sorted by engagement):
{captions}

Identify:
1. The top 5 winning hook patterns (the opening lines/phrases that grab attention)
   - Quote the hook text
   - Name the pattern (e.g., "Contrarian statement", "Question hook", "Number-led")
   - Rate effectiveness (High/Medium/Low based on engagement)

2. The top CTA patterns used at the end of posts

Be specific — quote actual text from the posts.""")
    ])

    state["hooks_raw"] = response.content
    return state


async def identify_opportunities(state: AnalysisState) -> AnalysisState:
    """Identify whitespace opportunities — what competitors AREN'T doing."""
    request = state["request"]

    brand_context = f"""Brand tone: {request.brand_kit.tone_of_voice or 'not set'}
Target audience: {request.brand_kit.target_audience or 'not set'}
Objective: {request.brand_kit.objective or 'not set'}"""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT + grounding(request.persona, request.account_model)),
        HumanMessage(content=f"""Based on the competitor analysis AND the brand's own voice (retrieved from their existing posts), identify whitespace opportunities — content gaps and underserved areas this specific brand could own.

FORMATS ANALYSIS:
{state['formats_raw']}

HOOKS/CTA ANALYSIS:
{state['hooks_raw']}

BRAND CONTEXT:
{brand_context}

{state['brand_voice_context']}

Identify 3-5 whitespace opportunities where this brand could differentiate:
- What topics/formats are competitors NOT covering?
- What audience needs are underserved?
- What content angles would stand out?

For each opportunity:
1. Name the area
2. Explain reasoning (why this is an opportunity)
3. Rate confidence (0.0 to 1.0)""")
    ])

    state["opportunities_raw"] = response.content
    return state


async def synthesize(state: AnalysisState) -> AnalysisState:
    """Combine all analysis into the final structured response."""
    response = await llm.ainvoke([
        SystemMessage(content="""You are a data synthesis engine. Convert raw analysis text into a structured JSON format. Be precise and faithful to the source analysis. Output ONLY valid JSON, no markdown."""),
        HumanMessage(content=f"""Convert the following analyses into this exact JSON structure:

{{
  "dominant_formats": [
    {{"format": "...", "frequency": "...", "avg_engagement": "...", "examples": ["...", "..."]}}
  ],
  "winning_hooks": [
    {{"hook_text": "...", "pattern": "...", "effectiveness": "High|Medium|Low"}}
  ],
  "whitespace_opportunities": [
    {{"area": "...", "reasoning": "...", "confidence": 0.8}}
  ],
  "content_cadence": "...",
  "key_takeaways": ["...", "...", "..."]
}}

FORMATS ANALYSIS:
{state['formats_raw']}

HOOKS/CTA ANALYSIS:
{state['hooks_raw']}

OPPORTUNITIES ANALYSIS:
{state['opportunities_raw']}

Return ONLY the JSON object. Include 3-5 items in each array. key_takeaways should be 3-5 concise bullet points summarizing the most important findings.""")
    ])

    import json
    raw = response.content.strip()
    # Handle potential markdown code blocks
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    parsed = json.loads(raw)
    state["final_result"] = AnalysisResponse(**parsed)
    return state


def build_analysis_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("gather_rag_context", gather_rag_context)
    graph.add_node("analyze_formats", analyze_formats)
    graph.add_node("analyze_hooks_ctas", analyze_hooks_ctas)
    graph.add_node("identify_opportunities", identify_opportunities)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("gather_rag_context")
    graph.add_edge("gather_rag_context", "analyze_formats")
    graph.add_edge("analyze_formats", "analyze_hooks_ctas")
    graph.add_edge("analyze_hooks_ctas", "identify_opportunities")
    graph.add_edge("identify_opportunities", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


analysis_graph = build_analysis_graph()


async def run_market_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """Run the full market analysis pipeline."""
    logger.info(f"Starting market analysis for project {request.project_id} with {len(request.competitors)} competitors and {len(request.signals)} signals")

    initial_state: AnalysisState = {
        "request": request,
        "market_context": "",
        "brand_voice_context": "",
        "formats_raw": "",
        "hooks_raw": "",
        "opportunities_raw": "",
        "final_result": None,
    }

    result = await analysis_graph.ainvoke(initial_state)

    if result["final_result"] is None:
        raise ValueError("Analysis pipeline failed to produce a result")

    logger.info(f"Market analysis complete for project {request.project_id}")
    return result["final_result"]
