"""Parse CrispASR stderr progress lines into structured progress data."""

import re

# CrispASR emits progress lines like:
#   progress:  45%|###       | 12/26 [00:03<00:04, 3.5 it/s]
#   crispasr: progress = 0.450
#   [50%]
#   Processing: 12/26 segments

_PERCENT_RE = re.compile(r"(\d+)%")
_PROGRESS_FLOAT_RE = re.compile(r"progress\s*[=:]\s*([\d.]+)")
_FRACTION_RE = re.compile(r"(\d+)/(\d+)")
_RTF_RE = re.compile(r"RTF[=:\s]+([\d.]+)")
_WPS_RE = re.compile(r"WPS[=:\s]+([\d.]+)")


def parse_progress_line(line):
    """Parse a stderr line for progress info.

    Returns a dict with any of:
        progress: float 0.0-1.0
        rtf: float (real-time factor)
        wps: float (words per second)

    Returns empty dict if no progress info found.
    """
    result = {}

    # Try progress = 0.45 format
    m = _PROGRESS_FLOAT_RE.search(line)
    if m:
        val = float(m.group(1))
        result["progress"] = val if val <= 1.0 else val / 100.0
        return _add_metrics(result, line)

    # Try 45% format
    m = _PERCENT_RE.search(line)
    if m:
        result["progress"] = int(m.group(1)) / 100.0
        return _add_metrics(result, line)

    # Try 12/26 fraction format
    m = _FRACTION_RE.search(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            result["progress"] = done / total
            return _add_metrics(result, line)

    return _add_metrics(result, line)


def _add_metrics(result, line):
    """Extract RTF and WPS if present."""
    m = _RTF_RE.search(line)
    if m:
        result["rtf"] = float(m.group(1))
    m = _WPS_RE.search(line)
    if m:
        result["wps"] = float(m.group(1))
    return result
