"""
Turn a reference the user shared (photo, PDF, text, video) into text the Studio
agent can read: image description by a vision model, PDF text, a Whisper
transcript for video/audio — plus a short brief on what is worth reusing.
"""
import asyncio
import base64
import io
import logging
import os

import fitz  # PyMuPDF: renders PDF pages to images for OCR
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


def _pdf_text(data: bytes, max_pages: int = MAX_PDF_PAGES) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages[:max_pages]
    parts = [(page.extract_text() or "").strip() for page in pages]
    return "\n\n".join(p for p in parts if p), len(reader.pages)


# ── OCR for PDFs made of images (Canva exports, scans): every page is rendered and
# read by the vision model, which keeps Arabic, Latin and digits as written. ──
OCR_MODEL = os.getenv("OCR_MODEL", VISION_MODEL)
OCR_CONCURRENCY = 4
OCR_MAX_PAGES_REFERENCE = 20  # a Studio reference; a library import reads the whole file
OCR_MIN_CHARS_PER_PAGE = 60  # below this the "text" is page numbers, not content
OCR_PROMPT = (
    "Transcribe every piece of text on this page exactly as written, in its original language and script "
    "(Arabic letters, Latin letters, digits, emojis). Keep the reading order and the line breaks, one line "
    "of output per line of text. Do not translate, do not summarize, do not describe images, do not add "
    "anything. If the page has no text at all, answer exactly: [EMPTY]"
)


def _render_pages(data: bytes, max_pages: int) -> list[bytes]:
    doc = fitz.open(stream=data, filetype="pdf")
    images: list[bytes] = []
    for index, page in enumerate(doc):
        if index >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        images.append(pix.tobytes("png"))
    return images


async def _ocr_page(client: AsyncOpenAI, png: bytes, index: int, sem: asyncio.Semaphore) -> tuple[int, str]:
    async with sem:
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        try:
            response = await client.chat.completions.create(
                model=OCR_MODEL,
                max_tokens=4000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    }
                ],
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning(f"OCR failed on page {index + 1}: {exc}")
            text = ""
    return index, "" if text == "[EMPTY]" else text


async def _ocr_pdf(client: AsyncOpenAI, data: bytes, max_pages: int) -> str:
    images = _render_pages(data, max_pages)
    sem = asyncio.Semaphore(OCR_CONCURRENCY)
    results = await asyncio.gather(*(_ocr_page(client, png, i, sem) for i, png in enumerate(images)))
    results.sort(key=lambda r: r[0])
    return "\n\n".join(text for _, text in results if text)


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
        page_cap = 1000 if request.full else MAX_PDF_PAGES
        text, pages = _pdf_text(data, max_pages=page_cap)
        meta["pages"] = pages
        read_pages = max(1, min(pages, page_cap))
        if len(text) < OCR_MIN_CHARS_PER_PAGE * read_pages:
            # Images instead of text: read the pages with the vision model
            ocr_cap = 1000 if request.full else OCR_MAX_PAGES_REFERENCE
            logger.info(f"Reference {request.reference_id}: PDF has no text layer, OCR on {min(pages, ocr_cap)} page(s)")
            text = await _ocr_pdf(client, data, ocr_cap)
            meta["ocr"] = OCR_MODEL
            if not text:
                raise ValueError("No readable text was found in this PDF, even with OCR. Paste the text instead.")
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
        # The backend scraped caption + metrics; we add the transcript when there is a video
        text = (request.text or "").strip()
        if request.video_url:
            try:
                data = await _download(request.video_url)
                if len(data) > WHISPER_MAX_BYTES:
                    meta["transcript_skipped"] = "video larger than 25 MB"
                else:
                    extension = EXTENSION_BY_MIME.get((request.mime_type or "").lower(), ".mp4")
                    transcript = await _transcribe(client, data, f"reference{extension}")
                    if transcript:
                        text = f"{text}\n\nTranscript:\n{transcript}"
                        meta["transcribed_by"] = "whisper-1"
            except Exception as exc:  # a missing transcript must not lose the caption
                logger.warning(f"Transcript failed for {request.reference_id}: {exc}")
                meta["transcript_skipped"] = str(exc)[:200]
        if not text:
            raise ValueError("Nothing readable was found in this post.")
    else:
        raise ValueError(f"Unsupported reference kind: {kind}")

    # A Studio reference is capped; a library import keeps the whole document
    text = text[: (3_000_000 if request.full else MAX_TEXT_CHARS)]
    summary = await _summarize(client, kind, request.title, text)
    logger.info(f"Reference {request.reference_id} ({kind}) extracted: {len(text)} chars")
    return ReferenceExtractResponse(extracted_text=text, summary=summary, meta=meta)
