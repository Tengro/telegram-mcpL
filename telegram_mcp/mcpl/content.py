"""Telegram media ↔ McplContentBlock conversion.

Conversion rules:
  - Text                     → McplTextContent
  - Photo / image document   → McplImageContent (inline base64) when within
                               the size cap, so the agent's model actually
                               *sees* the image. Falls back to a placeholder
                               + message URI when too large or undownloadable.
  - Small text document       → McplTextContent with the file contents inlined,
                               so the agent can read shared text files.
  - Other media (voice, video,
    pdf, binary, …)           → McplTextContent placeholder ("[<kind>]") plus a
                               McplResourceContent pointing at the message URI,
                               which the agent can follow with the media tools.

Media is downloaded into memory on the incoming path and base64-inlined so it
survives the host's `channels/incoming` → membrane → Anthropic conversion
(the host carries `image`/`text` blocks but flattens `resource` blocks to a
placeholder string, so only inlined media is actually visible to the model).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from .types import (
    McplContentBlock,
    McplImageContent,
    McplResourceContent,
    McplTextContent,
)

log = logging.getLogger("telegram_mcp.mcpl")

# Anthropic rejects images whose base64 payload exceeds ~5MB; base64 inflates
# raw bytes by ~4/3, so cap the raw download well under that.
IMAGE_INLINE_LIMIT_BYTES = 3_500_000
"""Max raw image bytes to inline as a base64 `image` block."""

TEXT_INLINE_LIMIT_BYTES = 256 * 1024
"""Max raw bytes of a text document to inline as a `text` block."""

# Image formats the Anthropic vision path accepts. Static stickers ride in as
# image/webp documents; animated (.tgs) / video (webm) stickers do not and fall
# back to a placeholder.
SUPPORTED_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)

# Document MIME types (beyond the text/* family) safe to decode as UTF-8 text.
TEXT_DOCUMENT_MIMES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
        "application/x-sh",
        "application/csv",
    }
)

# Fallback when the MIME type is generic (e.g. application/octet-stream) but the
# filename clearly denotes a text file.
TEXT_FILE_EXTENSIONS = (
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".html", ".htm", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".py", ".rs", ".go", ".java", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".rb", ".php", ".sh", ".bash", ".zsh", ".sql", ".lua", ".pl", ".r",
)


def message_uri(account_label: str, chat_id: int | str, message_id: int) -> str:
    """Stable URI for a single Telegram message.

    Used as `telegram://message/{account}/{chat}/{message}` so the agent
    can pass it to the existing fetch tools (or, eventually, an MCP
    resources/read endpoint) when it needs media we didn't inline.
    """
    return f"telegram://message/{account_label}/{chat_id}/{message_id}"


def media_kind(message: Any) -> str | None:
    """Return a short human label for the message's media type, if any."""
    media = getattr(message, "media", None)
    if media is None:
        return None
    cls = type(media).__name__
    # Telethon types: MessageMediaPhoto, MessageMediaDocument, MessageMediaWebPage,
    # MessageMediaContact, MessageMediaPoll, MessageMediaGeo, etc.
    mapping = {
        "MessageMediaPhoto": "photo",
        "MessageMediaDocument": "document",
        "MessageMediaWebPage": "link",
        "MessageMediaContact": "contact",
        "MessageMediaPoll": "poll",
        "MessageMediaGeo": "location",
        "MessageMediaVenue": "location",
        "MessageMediaDice": "dice",
        "MessageMediaGame": "game",
        "MessageMediaInvoice": "invoice",
        "MessageMediaUnsupported": "unsupported",
    }
    short = mapping.get(cls, cls.removeprefix("MessageMedia").lower() or "media")
    if short == "document":
        # Voice notes and round videos are documents with attribute flags
        document = getattr(media, "document", None)
        if document is not None:
            attributes = getattr(document, "attributes", []) or []
            for attr in attributes:
                attr_cls = type(attr).__name__
                if attr_cls == "DocumentAttributeAudio" and getattr(attr, "voice", False):
                    return "voice"
                if attr_cls == "DocumentAttributeVideo":
                    return "video"
                if attr_cls == "DocumentAttributeAnimated":
                    return "gif"
                if attr_cls == "DocumentAttributeSticker":
                    return "sticker"
    return short


def _document_info(media: Any) -> tuple[str | None, str | None, int | None]:
    """Return (mime_type, filename, size) for a MessageMediaDocument."""
    document = getattr(media, "document", None)
    if document is None:
        return None, None, None
    mime = getattr(document, "mime_type", None)
    size = getattr(document, "size", None)
    filename: str | None = None
    for attr in getattr(document, "attributes", []) or []:
        if type(attr).__name__ == "DocumentAttributeFilename":
            filename = getattr(attr, "file_name", None)
            break
    return mime, filename, size


def _is_text_document(mime: str | None, filename: str | None) -> bool:
    if mime:
        if mime.startswith("text/") or mime in TEXT_DOCUMENT_MIMES:
            return True
    if filename and filename.lower().endswith(TEXT_FILE_EXTENSIONS):
        return True
    return False


async def _download_bytes(message: Any) -> bytes | None:
    """Download a message's media into memory, or None if unavailable.

    Never raises — a download failure must degrade to the placeholder path,
    not tear down the event handler.
    """
    downloader = getattr(message, "download_media", None)
    if downloader is None:
        return None
    try:
        data = await downloader(file=bytes)
    except Exception:  # noqa: BLE001 — any failure → placeholder fallback
        log.exception(
            "media download failed for message %s", getattr(message, "id", "?")
        )
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return None


async def _media_to_rich_block(message: Any) -> McplContentBlock | None:
    """Try to turn a message's media into an inlined image/text block.

    Returns None when the media isn't a supported inline type, is too large,
    or can't be downloaded — the caller then emits the placeholder + URI.
    """
    media = getattr(message, "media", None)
    if media is None:
        return None
    cls = type(media).__name__

    # Photos — Telegram serves these as JPEG.
    if cls == "MessageMediaPhoto":
        data = await _download_bytes(message)
        if data is None or len(data) > IMAGE_INLINE_LIMIT_BYTES:
            return None
        return McplImageContent(
            type="image",
            data=base64.b64encode(data).decode("ascii"),
            mimeType="image/jpeg",
        )

    if cls == "MessageMediaDocument":
        mime, filename, size = _document_info(media)

        # Images shipped as documents (incl. static webp stickers).
        if mime in SUPPORTED_IMAGE_MIMES:
            if size is not None and size > IMAGE_INLINE_LIMIT_BYTES:
                return None
            data = await _download_bytes(message)
            if data is None or len(data) > IMAGE_INLINE_LIMIT_BYTES:
                return None
            return McplImageContent(
                type="image",
                data=base64.b64encode(data).decode("ascii"),
                mimeType=mime,
            )

        # Small text files — inline the contents so the agent can read them.
        if _is_text_document(mime, filename):
            if size is not None and size > TEXT_INLINE_LIMIT_BYTES:
                return None
            data = await _download_bytes(message)
            if data is None or len(data) > TEXT_INLINE_LIMIT_BYTES:
                return None
            body = data.decode("utf-8", errors="replace")
            label = filename or "text file"
            return McplTextContent(type="text", text=f"[file: {label}]\n{body}")

    return None


async def message_to_content_blocks(
    message: Any,
    *,
    account_label: str,
    chat_id: int | str,
) -> list[McplContentBlock]:
    """Convert a Telethon Message into MCPL content blocks.

    Text is emitted as a text block. Images and small text files are
    downloaded and inlined (base64 image / decoded text) so the model can
    perceive them directly. Any other media — or media too large to inline —
    degrades to a `[<kind>]` placeholder plus a McplResourceContent pointing
    at the message URI, which the agent can follow with the media tools.
    """
    blocks: list[McplContentBlock] = []
    text = getattr(message, "message", None) or ""
    kind = media_kind(message)

    if text:
        blocks.append(McplTextContent(type="text", text=text))

    if kind:
        rich = await _media_to_rich_block(message)
        if rich is not None:
            blocks.append(rich)
        else:
            # Couldn't inline — surface a placeholder (if there's no caption)
            # and the resource pointer so the agent can fetch on demand.
            if not text:
                blocks.append(McplTextContent(type="text", text=f"[{kind}]"))
            blocks.append(
                McplResourceContent(
                    type="resource",
                    uri=message_uri(account_label, chat_id, message.id),
                )
            )

    if not blocks:
        # Fallback for service messages with neither text nor media (joins,
        # pin updates, etc.). The agent at least sees something.
        blocks.append(McplTextContent(type="text", text="[message]"))

    return blocks
