"""
Turn a reference the user shared (photo, PDF, text, video) into text the Studio
agent can read: image description by a vision model, PDF text, a Whisper
transcript for video/audio — plus a short brief on what is worth reusing.
"""
import io
import logging
import os

import httpx
from openai import AsyncOpenAI
from pypdf import PdfReader

from schemas.references import ReferenceExtractRequest, ReferenceExtractResponse

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 40_000
MAX_PDF_PAGES = 60
WHISPER_MAX_BYTES = 25 * 1024 * 1024
SUMMARY_MODEL = os.getenv("REFERENCE_SUMMARY_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("STUDIO_MODEL", "gpt-4o")

EXTENSION_BY_MIME = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-m4v": ".m4v",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
}


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def _describe_image(client: AsyncOpenAI, url: str) -> str:
    response = await client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image for a social-media copywriter: subject, setting, mood, colours, "
                            "composition, any text on it, and the style it suggests (raw, polished, meme, editorial). "
                            "Five to eight concrete sentences, no preamble."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def _transcribe(client: AsyncOpenAI, data: bytes, filename: str) -> str:
    response = await client.audio.transcriptions.create(model="whisper-1", file=(filename, data))
    return (response.text or "").strip()


def _pdf_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages[:MAX_PDF_PAGES]
    parts = [(page.extract_text() or "").strip() for page in pages]
    return "\n\n".join(p for p in parts if p), len(reader.pages)


async def _summarize(client: AsyncOpenAI, kind: str, title: str | None, text: str) -> str:
    if not text.strip():
        return ""
    response = await client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=0.2,
        max_tokens=220,
        messages=[
            {
                "role": "system",
                "content": (
                    "You brief a social-media copywriter on a reference the user shared. In two to four sentences: "
                    "what it is, its structure (hook, body, CTA if any), its tone, and what is worth reusing. No preamble."
                ),
            },
            {"role": "user", "content": f"Kind: {kind}\nTitle: {title or 'untitled'}\n\nCONTENT:\n{text[:8000]}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def extract_reference(request: ReferenceExtractRequest) -> ReferenceExtractResponse:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kind = request.kind
    meta: dict = {}
    text = ""

    if kind == "image":
        text = await _describe_image(client, request.url)
        meta["described_by"] = VISION_MODEL
    elif kind == "pdf":
        data = await _download(request.url)
        text, pages = _pdf_text(data)
        meta["pages"] = pages
        if not text:
            raise ValueError("This PDF has no selectable text (scanned pages) — paste the text instead.")
    elif kind == "text":
        data = await _download(request.url)
        text = data.decode("utf-8", errors="ignore")
    elif kind in ("video", "audio"):
        data = await _download(request.url)
        if len(data) > WHISPER_MAX_BYTES:
            raise ValueError("Transcription accepts files up to 25 MB — upload a shorter or more compressed clip.")
        extension = EXTENSION_BY_MIME.get((request.mime_type or "").lower(), ".mp4" if kind == "video" else ".mp3")
        text = await _transcribe(client, data, f"reference{extension}")
        meta["transcribed_by"] = "whisper-1"
        if not text:
            raise ValueError("No speech was detected in this clip.")
    elif kind == "link":
        raise ValueError("Links are not read yet — upload the file itself or paste its text.")
    else:
        raise ValueError(f"Unsupported reference kind: {kind}")

    text = text[:MAX_TEXT_CHARS]
    summary = await _summarize(client, kind, request.title, text)
    logger.info(f"Reference {request.reference_id} ({kind}) extracted: {len(text)} chars")
    return ReferenceExtractResponse(extracted_text=text, summary=summary, meta=meta)
