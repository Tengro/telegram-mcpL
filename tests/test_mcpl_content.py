"""Phase 4 tests — Telegram message → McplContentBlock conversion."""

import base64
from types import SimpleNamespace

import pytest

from telegram_mcp.mcpl.content import (
    IMAGE_INLINE_LIMIT_BYTES,
    TEXT_INLINE_LIMIT_BYTES,
    media_kind,
    message_to_content_blocks,
    message_uri,
)


# ---------------------------------------------------------------------------
# Telethon media stand-ins. media_kind / the conversion key off the class
# *name*, so naming these to match Telethon's types is what matters.
# ---------------------------------------------------------------------------


class MessageMediaPhoto:
    pass


class DocumentAttributeFilename:
    def __init__(self, file_name):
        self.file_name = file_name


class DocumentAttributeSticker:
    pass


class FakeDocument:
    def __init__(self, *, mime_type=None, size=None, attributes=None):
        self.mime_type = mime_type
        self.size = size
        self.attributes = attributes or []


class MessageMediaDocument:
    def __init__(self, document):
        self.document = document


class FakeMessage:
    """Minimal async-downloadable Message double."""

    def __init__(self, *, message="", media=None, id=1, data=b""):
        self.message = message
        self.media = media
        self.id = id
        self._data = data

    async def download_media(self, file=bytes):
        return self._data


# ---------------------------------------------------------------------------
# media_kind / message_uri (unchanged behavior)
# ---------------------------------------------------------------------------


def test_message_uri_format():
    assert message_uri("default", 1234, 567) == "telegram://message/default/1234/567"
    # Negative chat IDs (channels in Telegram) survive verbatim
    assert message_uri("workacct", -100123, 999) == "telegram://message/workacct/-100123/999"


def test_media_kind_no_media():
    assert media_kind(SimpleNamespace(media=None)) is None


def test_media_kind_photo():
    assert media_kind(SimpleNamespace(media=MessageMediaPhoto())) == "photo"


def test_media_kind_voice_note():
    class DocumentAttributeAudio:
        def __init__(self, voice):
            self.voice = voice

    doc = FakeDocument(attributes=[DocumentAttributeAudio(voice=True)])
    assert media_kind(SimpleNamespace(media=MessageMediaDocument(doc))) == "voice"


def test_media_kind_sticker():
    doc = FakeDocument(attributes=[DocumentAttributeSticker()])
    assert media_kind(SimpleNamespace(media=MessageMediaDocument(doc))) == "sticker"


# ---------------------------------------------------------------------------
# Text / fallback paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only():
    msg = FakeMessage(message="hello world", media=None, id=42)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=100)
    assert blocks == [{"type": "text", "text": "hello world"}]


@pytest.mark.asyncio
async def test_empty_falls_back():
    msg = FakeMessage(message="", media=None, id=42)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=100)
    assert blocks == [{"type": "text", "text": "[message]"}]


@pytest.mark.asyncio
async def test_undownloadable_media_yields_placeholder_and_resource():
    # SimpleNamespace has no download_media → graceful fallback to placeholder.
    msg = SimpleNamespace(message="", media=MessageMediaPhoto(), id=42)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=100)
    assert blocks == [
        {"type": "text", "text": "[photo]"},
        {"type": "resource", "uri": "telegram://message/default/100/42"},
    ]


# ---------------------------------------------------------------------------
# Image inlining
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_photo_inlined_as_image_block():
    raw = b"\xff\xd8\xff\xe0fakejpegbytes"
    msg = FakeMessage(message="look at this", media=MessageMediaPhoto(), id=42, data=raw)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=100)
    assert blocks[0] == {"type": "text", "text": "look at this"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["mimeType"] == "image/jpeg"
    assert base64.b64decode(blocks[1]["data"]) == raw
    # No resource pointer once we've inlined the image.
    assert all(b["type"] != "resource" for b in blocks)


@pytest.mark.asyncio
async def test_photo_only_no_caption_inlines_image():
    raw = b"jpegjpeg"
    msg = FakeMessage(message="", media=MessageMediaPhoto(), id=7, data=raw)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=5)
    assert blocks == [
        {"type": "image", "data": base64.b64encode(raw).decode("ascii"), "mimeType": "image/jpeg"}
    ]


@pytest.mark.asyncio
async def test_oversize_photo_falls_back_to_placeholder():
    big = b"x" * (IMAGE_INLINE_LIMIT_BYTES + 1)
    msg = FakeMessage(message="", media=MessageMediaPhoto(), id=9, data=big)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=5)
    assert blocks == [
        {"type": "text", "text": "[photo]"},
        {"type": "resource", "uri": "telegram://message/default/5/9"},
    ]


@pytest.mark.asyncio
async def test_webp_sticker_document_inlined_as_image():
    raw = b"RIFF....WEBP"
    doc = FakeDocument(
        mime_type="image/webp", size=len(raw), attributes=[DocumentAttributeSticker()]
    )
    msg = FakeMessage(media=MessageMediaDocument(doc), id=3, data=raw)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=8)
    assert blocks == [
        {"type": "image", "data": base64.b64encode(raw).decode("ascii"), "mimeType": "image/webp"}
    ]


@pytest.mark.asyncio
async def test_oversize_image_document_skipped_by_declared_size():
    # Declared size over cap → we don't even download; placeholder fallback.
    doc = FakeDocument(mime_type="image/png", size=IMAGE_INLINE_LIMIT_BYTES + 1)
    msg = FakeMessage(media=MessageMediaDocument(doc), id=11, data=b"unused")
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=8)
    assert blocks == [
        {"type": "text", "text": "[document]"},
        {"type": "resource", "uri": "telegram://message/default/8/11"},
    ]


# ---------------------------------------------------------------------------
# Text-file inlining
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_document_inlined_with_filename_label():
    body = b"line one\nline two\n"
    doc = FakeDocument(
        mime_type="text/plain", size=len(body),
        attributes=[DocumentAttributeFilename("notes.txt")],
    )
    msg = FakeMessage(media=MessageMediaDocument(doc), id=4, data=body)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=8)
    assert blocks == [
        {"type": "text", "text": "[file: notes.txt]\nline one\nline two\n"}
    ]


@pytest.mark.asyncio
async def test_text_document_detected_by_extension_when_mime_generic():
    body = b'{"k": 1}'
    doc = FakeDocument(
        mime_type="application/octet-stream", size=len(body),
        attributes=[DocumentAttributeFilename("data.json")],
    )
    msg = FakeMessage(media=MessageMediaDocument(doc), id=6, data=body)
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=8)
    assert blocks == [{"type": "text", "text": '[file: data.json]\n{"k": 1}'}]


@pytest.mark.asyncio
async def test_oversize_text_document_falls_back():
    doc = FakeDocument(
        mime_type="text/plain", size=TEXT_INLINE_LIMIT_BYTES + 1,
        attributes=[DocumentAttributeFilename("huge.log")],
    )
    msg = FakeMessage(media=MessageMediaDocument(doc), id=12, data=b"unused")
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=8)
    assert blocks == [
        {"type": "text", "text": "[document]"},
        {"type": "resource", "uri": "telegram://message/default/8/12"},
    ]


@pytest.mark.asyncio
async def test_binary_document_stays_placeholder():
    doc = FakeDocument(
        mime_type="application/pdf", size=2048,
        attributes=[DocumentAttributeFilename("report.pdf")],
    )
    msg = FakeMessage(media=MessageMediaDocument(doc), id=13, data=b"%PDF-1.7")
    blocks = await message_to_content_blocks(msg, account_label="default", chat_id=8)
    assert blocks == [
        {"type": "text", "text": "[document]"},
        {"type": "resource", "uri": "telegram://message/default/8/13"},
    ]
