# utils/format_utils.py
"""Time formatting utilities"""


def time_to_srt(seconds):
    """Convert time in seconds to SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def time_to_vtt(seconds):
    """Convert time in seconds to VTT format (HH:MM:SS.mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_time(time_str):
    """Format time string to consistent format"""
    if not time_str:
        return ""
    try:
        time_str = time_str.replace(",", ".")
        return f"{float(time_str):.3f}"
    except ValueError:
        return None
