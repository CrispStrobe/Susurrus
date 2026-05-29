# gui/widgets/__init__.py
"""GUI Widgets"""

from .advanced_options import AdvancedOptionsBox
from .collapsible_box import CollapsibleBox
from .crispasr_advanced_settings import CrispASRAdvancedSettingsBox
from .diarization_settings import DiarizationSettingsBox
from .translation_settings import TranslationSettingsWidget
from .tts_settings import TTSSettingsWidget
from .voxtral_settings import VoxtralSettingsBox

__all__ = [
    "AdvancedOptionsBox",
    "CollapsibleBox",
    "CrispASRAdvancedSettingsBox",
    "DiarizationSettingsBox",
    "TTSSettingsWidget",
    "TranslationSettingsWidget",
    "VoxtralSettingsBox",
]
