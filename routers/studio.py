from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.studio import StudioChatRequest
from agents.studio import stream_studio_turn

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/studio/chat")
async def studio_chat(request: StudioChatRequest):
    """One assistant turn of the Studio conversation, streamed as Server-Sent Events."""
    return StreamingResponse(
        stream_studio_turn(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
