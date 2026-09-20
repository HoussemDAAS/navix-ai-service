import logging
import os

from dotenv import load_dotenv

# IMPORTANT: load .env BEFORE importing routers/agents — they instantiate
# ChatOpenAI at module level and need OPENAI_API_KEY present.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analysis, directions, drafts, brand_memory, brand_kit, studio

log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Navix AI Service",
    description="LangGraph-powered AI agents for market analysis, content generation, and brand memory",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(analysis.router)
app.include_router(directions.router)
app.include_router(drafts.router)
app.include_router(brand_memory.router)
app.include_router(brand_kit.router)
app.include_router(studio.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "navix-ai"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
