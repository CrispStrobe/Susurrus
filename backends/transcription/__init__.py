"""Transcription backends"""

from .voxtral_api import VoxtralAPI
from .voxtral_local import VoxtralLocal

__all__ = ["VoxtralAPI", "VoxtralLocal"]
