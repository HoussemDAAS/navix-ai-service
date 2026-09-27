"""
Content understanding: what a scraped post actually says and shows.

Videos are transcribed with Whisper (the audio track first, then the video
file), images and carousels are described by a vision model, and a video with
no speech gets its cover described instead. The backend stores the result on
the post and re-embeds it, so the market analysis, the dossiers and the Studio
reason from real content rather than from captions alone.
"""
import asyncio
import logging
import os

import httpx
from openai import AsyncOpenAI

from schemas.content import ContentItem, ContentUnderstandRequest, ContentUnderstandResponse, ContentUnderstanding

logger = logging.getLogger(__name__)

VISION_MODEL = os.getenv("CONTENT_VISION_MODEL", "gpt-4o")
WHISPER_MAX_BYTES = 25 * 1024 * 1024
MAX_VIDEO_SECONDS = 20 * 60
CONCURRENCY = int(os.getenv("CONTENT_UNDERSTAND_CONCURRENCY", "4"))
MIN_SPEECH_CHARS = 20
VIDEO_TYPES = {"video", "reel", "reels", "short", "shorts", "tiktok", "clip"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MEDIA_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4a", ".mp3", ".wav", ".ogg", ".m4v", ".aac")


def _headers(platform: str) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if platform == "tiktok":
        headers["Referer"] = "https://www.tiktok.com/"
    elif platform == "instagram":
        headers["Referer"] = "https://www.instagram.com/"
    return headers


async def _download_capped(url: str, platform: str) -> bytes | None:
    """Download a media file, giving up as soon as it passes Whisper's size limit."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True, headers=_headers(platform)
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > WHISPER_MAX_BYTES:
                return None
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > WHISPER_MAX_BYTES:
                    return None
                chunks.append(chunk)
            return b"".join(chunks)


def _filename(url: str, fallback: str) -> str:
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1].lower()
    return f"media{ext}" if ext in MEDIA_EXTENSIONS else fallback


async def _transcribe(client: AsyncOpenAI, data: bytes, filename: str) -> tuple[str, str | None]:
    response = await client.audio.transcriptions.create(
        model="whisper-1", file=(filename, data), response_format="verbose_json"
    )
    text = (getattr(response, "text", "") or "").strip()
    language = getattr(response, "language", None)
    return text, language


async def _describe(client: AsyncOpenAI, url: str, item: ContentItem, is_video: bool) -> str:
    if is_video:
        what = "the cover frame of a short video"
    elif (item.image_count or 0) > 1:
        what = f"the first image of a {item.image_count}-image carousel"
    else:
        what = "a social-media image post"
    prompt = (
        f"This is {what} on {item.platform}. For a social-media strategist, describe in 3 to 5 sentences: "
        "what is shown, any text written on the image (quote it exactly), the visual style (raw phone footage, "
        "studio, meme, text on background, product shot, screenshot) and what makes it stop a scroll. No preamble."
    )
    response = await client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=260,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def _understand_one(client: AsyncOpenAI, item: ContentItem) -> ContentUnderstanding:
    result = ContentUnderstanding(id=item.id)
    is_video = (item.content_type or "").lower() in VIDEO_TYPES or bool(item.video_url or item.audio_url)
    errors: list[str] = []

    if is_video and (item.duration_seconds or 0) <= MAX_VIDEO_SECONDS:
        # The audio track is a fraction of the video's size: try it first
        for url in [u for u in (item.audio_url, item.video_url) if u]:
            try:
                data = await _download_capped(url, item.platform)
                if data is None:
                    errors.append("media too large for transcription")
                    continue
                text, language = await _transcribe(client, data, _filename(url, "media.mp4"))
                result.transcript = text or None
                result.language = language
                break
            except Exception as exc:
                errors.append(f"transcription failed: {str(exc)[:120]}")

    # Images always get a description; a video only when nobody speaks in it
    needs_visual = bool(item.thumbnail_url) and (not is_video or len(result.transcript or "") < MIN_SPEECH_CHARS)
    if needs_visual:
        try:
            result.visual_summary = await _describe(client, item.thumbnail_url or "", item, is_video) or None
        except Exception as exc:
            errors.append(f"vision failed: {str(exc)[:120]}")

    if errors and not result.transcript and not result.visual_summary:
        result.error = "; ".join(errors)
    return result


async def understand_content(request: ContentUnderstandRequest) -> ContentUnderstandResponse:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120.0, max_retries=1)
    gate = asyncio.Semaphore(CONCURRENCY)

    async def guarded(item: ContentItem) -> ContentUnderstanding:
        async with gate:
            try:
                return await _understand_one(client, item)
            except Exception as exc:  # one bad post must not sink the batch
                return ContentUnderstanding(id=item.id, error=str(exc)[:200])

    results = await asyncio.gather(*(guarded(item) for item in request.items))
    done = [r for r in results if r.transcript or r.visual_summary]
    logger.info(f"Content understanding: {len(done)}/{len(results)} posts read for project {request.project_id}")
    return ContentUnderstandResponse(items=list(results))
