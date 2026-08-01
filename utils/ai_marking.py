"""Machine-readable AI-generation marking for synthesized audio.

EU AI Act Art. 50(2) requires providers of AI systems that generate synthetic
audio to mark the output in a machine-readable format. C2PA Content
Credentials (``utils.c2pa_signing``) are the primary mechanism, but that
degrades to a no-op when the optional ``c2pa-audio`` library is absent — which
would leave a default install emitting unmarked synthetic audio.

This module provides a dependency-free fallback in each container's own
native metadata format:

* **WAV** — a standard RIFF ``LIST/INFO`` chunk appended to the file.
* **MP3** — an ID3v2.4 tag prepended to the file (``TSSE``/``COMM``).

Any ordinary parser (ffmpeg, sox, soundfile, Windows Explorer) reads these,
and parsers that don't care skip them, so the audio payload stays
byte-identical for playback.

The layers are complementary, not alternatives: C2PA is cryptographically
verifiable, this one is merely declarative. Apply both when possible.

Handling MP3 matters because it is not a hypothetical: edge-tts synthesizes
MP3 natively, and the GUI's save dialog offers ``.mp3``. Marking only WAV
meant those paths emitted unmarked synthetic audio.
"""

import logging
import os
import shutil
import struct

logger = logging.getLogger(__name__)

# RIFF INFO tag IDs (see the RIFF spec's registered "INFO" list)
_TAG_SOFTWARE = b"ISFT"  # producing software
_TAG_COMMENT = b"ICMT"  # free-text comment
_TAG_ENGINEER = b"IENG"  # "engineer" — used here for the generating model
_TAG_TECHNICIAN = b"ITCH"  # technician — used here for the marking standard

#: Text written into ICMT. Kept stable so detectors can string-match it.
AI_DISCLOSURE_TEXT = (
    "AI-GENERATED AUDIO. Synthesized by an AI system. Marked per EU AI Act Art. 50(2)."
)

_MARKER_SENTINEL = "AI-GENERATED AUDIO"


def _iter_chunks(data):
    """Yield ``(chunk_id, start, size)`` for each top-level RIFF chunk.

    ``start`` is the offset of the chunk's payload. Malformed or truncated
    chunks end iteration rather than raising — callers treat a short read as
    "no marker found".
    """
    offset = 12  # past 'RIFF' <size> 'WAVE'
    end = len(data)
    while offset + 8 <= end:
        chunk_id = data[offset : offset + 4]
        (size,) = struct.unpack("<I", data[offset + 4 : offset + 8])
        payload = offset + 8
        if payload + size > end:
            return
        yield chunk_id, payload, size
        # RIFF chunks are word-aligned: odd sizes carry a pad byte.
        offset = payload + size + (size % 2)


def _is_wav(data):
    return len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"


def _build_info_chunk(entries):
    """Build a ``LIST`` chunk of type ``INFO`` from ``{tag: text}``."""
    body = b"INFO"
    for tag, text in entries.items():
        raw = text.encode("utf-8", errors="replace") + b"\x00"
        if len(raw) % 2:
            raw += b"\x00"  # pad subchunk to even length
        body += tag + struct.pack("<I", len(raw)) + raw
    return b"LIST" + struct.pack("<I", len(body)) + body


def embed_wav_ai_marker(wav_path, model=None, software="Susurrus"):
    """Append a machine-readable AI-generation marker to a WAV file in place.

    Args:
        wav_path: Path to the WAV file to mark.
        model: Optional model/backend identifier recorded in ``IENG``.
        software: Producing application name recorded in ``ISFT``.

    Returns:
        True if the marker was written or was already present, False if the
        file is missing, unreadable, or not a RIFF/WAVE file.
    """
    try:
        with open(wav_path, "rb") as f:
            header = f.read(12)
            if not _is_wav(header):
                logger.warning("AI marking skipped, not a RIFF/WAVE file: %s", wav_path)
                return False

        if read_wav_ai_marker(wav_path):
            logger.debug("AI marker already present: %s", wav_path)
            return True

        entries = {
            _TAG_SOFTWARE: software,
            _TAG_COMMENT: AI_DISCLOSURE_TEXT,
            _TAG_TECHNICIAN: "EU-AI-Act-Art50-2",
        }
        if model:
            entries[_TAG_ENGINEER] = str(model)

        chunk = _build_info_chunk(entries)

        # The marker goes at the end, so append in place rather than rewriting
        # the file. Synthesized audio can run to hundreds of megabytes, and
        # read-all/write-all cost that much memory and twice that much I/O to
        # add ~150 bytes.
        #
        # Order matters for crash-safety: the payload is written first and the
        # RIFF size field updated last. A file interrupted between the two is
        # still a valid WAV whose header simply does not count the trailing
        # bytes — readers ignore them. The reverse order would advertise a
        # length the file does not have.
        with open(wav_path, "r+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size % 2:
                f.write(b"\x00")  # RIFF chunks are word-aligned
                size += 1
            f.write(chunk)
            total = size + len(chunk)
            f.seek(4)
            f.write(struct.pack("<I", total - 8))
    except OSError as e:
        logger.warning("AI marking failed for %s: %s", wav_path, e)
        return False

    logger.info("AI marker embedded: %s", wav_path)
    return True


def _parse_info_chunk(body):
    """Decode a ``LIST/INFO`` chunk body into ``{tag: text}``."""
    entries = {}
    offset = 4  # past the "INFO" form type
    end = len(body)
    while offset + 8 <= end:
        tag = body[offset : offset + 4]
        (sub_size,) = struct.unpack("<I", body[offset + 4 : offset + 8])
        payload = offset + 8
        if payload + sub_size > end:
            break
        text = body[payload : payload + sub_size]
        entries[tag.decode("ascii", errors="replace")] = text.rstrip(b"\x00").decode(
            "utf-8", errors="replace"
        )
        offset = payload + sub_size + (sub_size % 2)
    return entries


def read_wav_ai_marker(wav_path, _data=None):
    """Read the AI-generation marker from a WAV file.

    Walks the chunk table by seeking over each chunk's header rather than
    loading the file. The marker lives at the very end of the file, so reading
    the whole thing to find ~150 bytes meant paying for the entire audio
    payload on every check — and it is checked before every write.

    Args:
        wav_path: Path to the WAV file to inspect.
        _data: Internal — pre-read file bytes, used by callers that already
            hold the content in memory.

    Returns:
        A dict of the decoded INFO tags (e.g. ``{"ICMT": ..., "IENG": ...}``)
        if an AI marker is present, otherwise None.
    """
    if _data is not None:
        if not _is_wav(_data):
            return None
        for chunk_id, start, size in _iter_chunks(_data):
            if chunk_id != b"LIST" or size < 4 or _data[start : start + 4] != b"INFO":
                continue
            entries = _parse_info_chunk(_data[start : start + size])
            if any(_MARKER_SENTINEL in v for v in entries.values()):
                return entries
        return None

    try:
        with open(wav_path, "rb") as f:
            if not _is_wav(f.read(12)):
                return None

            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            offset = 12

            while offset + 8 <= file_size:
                f.seek(offset)
                header = f.read(8)
                if len(header) < 8:
                    return None
                chunk_id = header[0:4]
                (size,) = struct.unpack("<I", header[4:8])
                payload = offset + 8
                if payload + size > file_size:
                    return None

                if chunk_id == b"LIST" and size >= 4:
                    body = f.read(size)
                    if body[0:4] == b"INFO":
                        entries = _parse_info_chunk(body)
                        if any(_MARKER_SENTINEL in v for v in entries.values()):
                            return entries

                offset = payload + size + (size % 2)
    except OSError:
        return None

    return None


def is_ai_marked(wav_path):
    """Return True if *wav_path* carries an AI-generation marker."""
    return read_ai_marker(wav_path) is not None


# --------------------------------------------------------------------------
# MP3 / ID3v2.4
# --------------------------------------------------------------------------

_ID3_TAG_SOFTWARE = b"TSSE"  # "software/hardware and settings used for encoding"
_ID3_TAG_COMMENT = b"COMM"
_ID3_TAG_ENCODED_BY = b"TENC"


def _syncsafe(n):
    """Encode *n* as a 4-byte syncsafe integer (7 bits per byte)."""
    return bytes(((n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F))


def _read_syncsafe(data):
    return (data[0] << 21) | (data[1] << 14) | (data[2] << 7) | data[3]


def _text_frame(frame_id, text):
    """Build a UTF-8 ID3v2.4 text frame."""
    payload = b"\x03" + text.encode("utf-8") + b"\x00"
    return frame_id + _syncsafe(len(payload)) + b"\x00\x00" + payload


def _comment_frame(text, description="AI disclosure"):
    """Build a UTF-8 ID3v2.4 COMM frame."""
    payload = (
        b"\x03" + b"eng" + description.encode("utf-8") + b"\x00" + text.encode("utf-8") + b"\x00"
    )
    return _ID3_TAG_COMMENT + _syncsafe(len(payload)) + b"\x00\x00" + payload


def _existing_id3_size(data):
    """Return the total byte length of a leading ID3v2 tag, or 0."""
    if len(data) < 10 or data[0:3] != b"ID3":
        return 0
    size = _read_syncsafe(data[6:10])
    footer = 10 if (data[5] & 0x10) else 0
    return 10 + size + footer


def _iter_id3_frames(tag):
    """Yield ``(frame_id, payload)`` for each frame in an ID3v2 tag body."""
    offset = 10
    end = len(tag)
    while offset + 10 <= end:
        frame_id = tag[offset : offset + 4]
        if frame_id == b"\x00\x00\x00\x00":
            return
        size = _read_syncsafe(tag[offset + 4 : offset + 8])
        payload = offset + 10
        if size <= 0 or payload + size > end:
            return
        yield frame_id, tag[payload : payload + size]
        offset = payload + size


def _decode_id3_text(payload, frame_id=None):
    """Decode an ID3 text/comment payload to a string, best effort.

    ``COMM`` carries a 3-byte language code and a null-terminated short
    description ahead of the actual text; both are stripped so callers see
    the comment itself rather than ``engAI disclosure\\x00...``.
    """
    if not payload:
        return ""
    encoding = payload[0]
    body = payload[1:]
    codec = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}.get(encoding, "utf-8")

    if frame_id == _ID3_TAG_COMMENT:
        body = body[3:]  # language
        sep = body.find(b"\x00")
        if sep != -1:
            body = body[sep + 1 :]  # short description

    try:
        text = body.decode(codec, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    return text.strip("\x00")


def embed_mp3_ai_marker(mp3_path, model=None, software="Susurrus"):
    """Prepend an ID3v2.4 AI-generation marker to an MP3, preserving frames.

    An MP3 may already carry an ID3v2 tag (edge-tts output does). Two stacked
    tags are not valid ID3 — only the first would be read — so an existing tag
    is parsed and its frames are carried into the rewritten tag.
    """
    # Only the existing tag is read, never the audio: an ID3v2 tag declares its
    # own length in the first 10 bytes, so there is no reason to pull a
    # multi-megabyte MP3 into memory to inspect ~1 KB of metadata.
    try:
        with open(mp3_path, "rb") as f:
            head = f.read(10)
            old_size = _existing_id3_size(head)
            existing = head + f.read(max(0, old_size - 10)) if old_size else b""
    except OSError as e:
        logger.warning("AI marking failed to read %s: %s", mp3_path, e)
        return False

    if existing and read_mp3_ai_marker(mp3_path, _data=existing):
        logger.debug("AI marker already present: %s", mp3_path)
        return True

    kept = b""
    if old_size:
        for frame_id, payload in _iter_id3_frames(existing):
            if frame_id in (_ID3_TAG_SOFTWARE, _ID3_TAG_COMMENT, _ID3_TAG_ENCODED_BY):
                continue  # replaced below
            kept += frame_id + _syncsafe(len(payload)) + b"\x00\x00" + payload

    frames = (
        kept
        + _text_frame(_ID3_TAG_SOFTWARE, software)
        + _text_frame(_ID3_TAG_ENCODED_BY, "EU-AI-Act-Art50-2")
        + _comment_frame(AI_DISCLOSURE_TEXT)
    )
    if model:
        frames += _text_frame(b"TXXX", f"model\x00{model}")

    tag = b"ID3" + b"\x04\x00" + b"\x00" + _syncsafe(len(frames)) + frames

    # A tag must sit at the front, so the audio has to move — but it can be
    # streamed rather than buffered whole.
    tmp_path = f"{mp3_path}.aimark.tmp"
    try:
        with open(mp3_path, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(tag)
            src.seek(old_size)
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        os.replace(tmp_path, mp3_path)
    except OSError as e:
        logger.warning("AI marking failed to write %s: %s", mp3_path, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False

    logger.info("AI marker embedded (ID3): %s", mp3_path)
    return True


def read_mp3_ai_marker(mp3_path, _data=None):
    """Read the AI-generation marker from an MP3's ID3v2 tag, or None.

    Reads only the tag, whose length is declared in its own header.
    """
    if _data is None:
        try:
            with open(mp3_path, "rb") as f:
                head = f.read(10)
                size = _existing_id3_size(head)
                if not size:
                    return None
                _data = head + f.read(max(0, size - 10))
        except OSError:
            return None

    size = _existing_id3_size(_data)
    if not size:
        return None

    entries = {}
    for frame_id, payload in _iter_id3_frames(_data[:size]):
        if frame_id in (_ID3_TAG_SOFTWARE, _ID3_TAG_COMMENT, _ID3_TAG_ENCODED_BY, b"TXXX"):
            entries[frame_id.decode("ascii", errors="replace")] = _decode_id3_text(
                payload, frame_id
            )

    if any(_MARKER_SENTINEL in v for v in entries.values()):
        return entries
    return None


# --------------------------------------------------------------------------
# Format dispatch
# --------------------------------------------------------------------------

#: Extensions this module can mark declaratively.
MARKABLE_EXTENSIONS = (".wav", ".mp3")


def embed_ai_marker(path, model=None, software="Susurrus"):
    """Mark *path* as AI-generated using its container's metadata format.

    Returns True if a marker was written or already present, False if the
    container is unsupported or the file could not be written.
    """
    if not path:
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return embed_wav_ai_marker(path, model=model, software=software)
    if ext == ".mp3":
        return embed_mp3_ai_marker(path, model=model, software=software)
    logger.warning(
        "No declarative AI marker for %s — EU AI Act Art. 50(2) marking of "
        "this container is not implemented; output may be unmarked.",
        path,
    )
    return False


def read_ai_marker(path):
    """Read the AI-generation marker from *path*, whatever its container."""
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return read_wav_ai_marker(path)
    if ext == ".mp3":
        return read_mp3_ai_marker(path)
    return None
