# gui/widgets/__init__.py
"""GUI Widgets"""

from .advanced_options import AdvancedOptionsBox
from .collapsible_box import CollapsibleBox
from .crispasr_advanced_settings import CrispASRAdvancedSettingsBox
from .diarization_settings import DiarizationSettingsBox
from .translation_settings import TranslationSettingsWidget
from .tts_settings import TTSSettingsWidget
from .voxtral_settings import VoxtralSettingsBox

# New widgets — lazy-imported to avoid PyQt6 crash when running headlessly
# from .history_panel import HistoryPanel
# from .batch_panel import BatchPanel
# from .waveform_widget import WaveformWidget
# from .log_viewer import LogViewer

__all__ = [
    "AdvancedOptionsBox",
    "BatchPanel",
    "CollapsibleBox",
    "CrispASRAdvancedSettingsBox",
    "DiarizationSettingsBox",
    "HistoryPanel",
    "LogViewer",
    "TTSSettingsWidget",
    "TranslationSettingsWidget",
    "VoxtralSettingsBox",
    "WaveformWidget",
]
