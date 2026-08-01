"""Machine-readable AI-generation marking for WAV files.

EU AI Act Art. 50(2) requires providers of AI systems that generate synthetic
audio to mark the output in a machine-readable format. C2PA Content
Credentials (``utils.c2pa_signing``) are the primary mechanism, but that
degrades to a no-op when the optional ``c2pa-audio`` library is absent — which
would leave a default install emitting unmarked synthetic audio.

This module provides a dependency-free fallback: a standard RIFF ``LIST/INFO``
metadata chunk appended to the WAV. Any RIFF parser (ffmpeg, sox, soundfile,
Windows Explorer) can read it, and parsers that don't care skip it, so the
audio stays byte-identical for playback.

The two layers are complementary, not alternatives: C2PA is cryptographically
verifiable, the INFO chunk is merely declarative. Apply both when possible.
"""

import logging
import os
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
            data = f.read()
    except OSError as e:
        logger.warning("AI marking failed to read %s: %s", wav_path, e)
        return False

    if not _is_wav(data):
        logger.warning("AI marking skipped, not a RIFF/WAVE file: %s", wav_path)
        return False

    if read_wav_ai_marker(wav_path, _data=data):
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

    # Appending grows the file, so the RIFF header size must grow with it.
    # An odd-length final chunk needs its pad byte before we append.
    padded = data + (b"\x00" if len(data) % 2 else b"")
    marked = padded + chunk
    marked = marked[:4] + struct.pack("<I", len(marked) - 8) + marked[8:]

    tmp_path = f"{wav_path}.aimark.tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(marked)
        os.replace(tmp_path, wav_path)
    except OSError as e:
        logger.warning("AI marking failed to write %s: %s", wav_path, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False

    logger.info("AI marker embedded: %s", wav_path)
    return True


def read_wav_ai_marker(wav_path, _data=None):
    """Read the AI-generation marker from a WAV file.

    Args:
        wav_path: Path to the WAV file to inspect.
        _data: Internal — pre-read file bytes, to avoid a second read.

    Returns:
        A dict of the decoded INFO tags (e.g. ``{"ICMT": ..., "IENG": ...}``)
        if an AI marker is present, otherwise None.
    """
    if _data is None:
        try:
            with open(wav_path, "rb") as f:
                _data = f.read()
        except OSError:
            return None

    if not _is_wav(_data):
        return None

    for chunk_id, start, size in _iter_chunks(_data):
        if chunk_id != b"LIST" or size < 4:
            continue
        if _data[start : start + 4] != b"INFO":
            continue

        entries = {}
        offset = start + 4
        end = start + size
        while offset + 8 <= end:
            tag = _data[offset : offset + 4]
            (sub_size,) = struct.unpack("<I", _data[offset + 4 : offset + 8])
            payload = offset + 8
            if payload + sub_size > end:
                break
            text = _data[payload : payload + sub_size]
            entries[tag.decode("ascii", errors="replace")] = text.rstrip(b"\x00").decode(
                "utf-8", errors="replace"
            )
            offset = payload + sub_size + (sub_size % 2)

        if any(_MARKER_SENTINEL in v for v in entries.values()):
            return entries

    return None


def is_ai_marked(wav_path):
    """Return True if *wav_path* carries an AI-generation marker."""
    return read_wav_ai_marker(wav_path) is not None
