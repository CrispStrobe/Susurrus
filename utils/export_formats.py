"""Export transcription segments to various formats (SRT, VTT, JSON, CSV, TXT)."""

import csv
import io
import json


def _fmt_ts_srt(seconds):
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ts_vtt(seconds):
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def export_srt(segments):
    """Export segments as SRT subtitle format.

    Args:
        segments: list of (start, end, text) tuples or dicts with start/end/text keys.

    Returns:
        SRT-formatted string.
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        start, end, text = _unpack_segment(seg)
        lines.append(str(i))
        lines.append(f"{_fmt_ts_srt(start)} --> {_fmt_ts_srt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_vtt(segments):
    """Export segments as WebVTT subtitle format.

    Args:
        segments: list of (start, end, text) tuples or dicts.

    Returns:
        VTT-formatted string.
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        start, end, text = _unpack_segment(seg)
        lines.append(f"{_fmt_ts_vtt(start)} --> {_fmt_ts_vtt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_json(segments, metadata=None):
    """Export segments as JSON.

    Args:
        segments: list of (start, end, text) tuples or dicts.
        metadata: optional dict with extra info (backend, model, language, etc.)

    Returns:
        JSON string.
    """
    data = {
        "segments": [
            {"start": start, "end": end, "text": text}
            for start, end, text in (_unpack_segment(s) for s in segments)
        ],
    }
    if metadata:
        data["metadata"] = metadata
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_csv(segments):
    """Export segments as CSV.

    Args:
        segments: list of (start, end, text) tuples or dicts.

    Returns:
        CSV string with header row.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["start", "end", "text"])
    for seg in segments:
        start, end, text = _unpack_segment(seg)
        writer.writerow([f"{start:.3f}", f"{end:.3f}", text])
    return output.getvalue()


def export_txt(segments):
    """Export segments as plain text (one line per segment, no timestamps).

    Args:
        segments: list of (start, end, text) tuples or dicts.

    Returns:
        Plain text string.
    """
    return "\n".join(_unpack_segment(seg)[2] for seg in segments)


def _unpack_segment(seg):
    """Unpack a segment from tuple or dict form."""
    if isinstance(seg, dict):
        return seg.get("start", 0.0), seg.get("end", 0.0), seg.get("text", "")
    return seg[0], seg[1], seg[2]


# Format name → (extension, export function)
EXPORT_FORMATS = {
    "TXT": (".txt", export_txt),
    "SRT": (".srt", export_srt),
    "VTT": (".vtt", export_vtt),
    "JSON": (".json", export_json),
    "CSV": (".csv", export_csv),
}
