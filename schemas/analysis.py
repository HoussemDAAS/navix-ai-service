from pydantic import BaseModel
from .common import CompetitorData, BrandKitData, SignalData


class FormatInsight(BaseModel):
    format: str
    frequency: str
    avg_engagement: str
    examples: list[str]


class HookInsight(BaseModel):
    hook_text: str
    pattern: str
    effectiveness: str


class Opportunity(BaseModel):
    area: str
    reasoning: str
    confidence: float


class AnalysisRequest(BaseModel):
    project_id: str
    competitors: list[CompetitorData]
    brand_kit: BrandKitData
    signals: list[SignalData]
    # Account type chosen at signup: creator | ecommerce | agency
    persona: str | None = None
    # Field analytics computed by the backend from scraped posts (self vs field)
    account_model: dict | None = None


class AnalysisResponse(BaseModel):
    dominant_formats: list[FormatInsight]
    winning_hooks: list[HookInsight]
    whitespace_opportunities: list[Opportunity]
    content_cadence: str
    key_takeaways: list[str]
