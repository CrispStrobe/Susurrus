"""Minimal i18n — string lookup with English fallback."""

import logging

logger = logging.getLogger(__name__)

_current_locale = "en"

TRANSLATIONS = {
    "en": {
        "app.title": "Susurrus",
        "app.subtitle": "Audio Transcription, TTS & Translation Suite",
        "tab.transcription": "Transcription",
        "tab.tts": "Text-to-Speech",
        "tab.translation": "Translation",
        "tab.history": "History",
        "btn.transcribe": "Transcribe",
        "btn.abort": "Abort",
        "btn.save": "Save",
        "btn.stream_mic": "Stream Mic",
        "btn.stop_mic": "Stop Mic",
        "btn.detect_watermark": "Detect Watermark",
        "btn.synthesize": "Synthesize",
        "btn.play": "Play",
        "btn.translate": "Translate",
        "btn.refresh": "Refresh",
        "btn.delete": "Delete",
        "btn.clear_all": "Clear All",
        "btn.add_files": "Add Files...",
        "btn.start_batch": "Start Batch",
        "btn.stop": "Stop",
        "btn.clear_done": "Clear Done",
        "label.metrics": "Metrics",
        "label.transcription": "Transcription",
        "label.backend": "Backend:",
        "label.model": "Model:",
        "label.device": "Device:",
        "label.language": "Language:",
        "label.voice": "Voice:",
        "label.output_file": "Output file:",
        "label.search": "Search history...",
        "label.batch_queue": "Batch Queue",
        "menu.file": "&File",
        "menu.tools": "&Tools",
        "menu.view": "&View",
        "menu.help": "&Help",
        "warn.no_watermark": (
            "Watermarking disabled. AI-content marking responsibility "
            "rests with the operator per EU AI Act Art. 50."
        ),
        "warn.no_transcription": "No transcription available to save.",
        "wizard.step1_title": "Step 1: Select Reference Audio",
        "wizard.step2_title": "Step 2: Reference Transcription",
        "wizard.step3_title": "Step 3: Confirm",
    },
    "de": {
        "app.title": "Susurrus",
        "app.subtitle": "Audio-Transkription, TTS & Übersetzungssuite",
        "tab.transcription": "Transkription",
        "tab.tts": "Sprachsynthese",
        "tab.translation": "Übersetzung",
        "tab.history": "Verlauf",
        "btn.transcribe": "Transkribieren",
        "btn.abort": "Abbrechen",
        "btn.save": "Speichern",
        "btn.stream_mic": "Mikrofon streamen",
        "btn.stop_mic": "Mikrofon stoppen",
        "btn.detect_watermark": "Wasserzeichen erkennen",
        "btn.synthesize": "Synthetisieren",
        "btn.play": "Abspielen",
        "btn.translate": "Übersetzen",
        "btn.refresh": "Aktualisieren",
        "btn.delete": "Löschen",
        "btn.clear_all": "Alle löschen",
        "btn.add_files": "Dateien hinzufügen...",
        "btn.start_batch": "Stapel starten",
        "btn.stop": "Stoppen",
        "btn.clear_done": "Erledigte löschen",
        "label.metrics": "Metriken",
        "label.transcription": "Transkription",
        "label.backend": "Backend:",
        "label.model": "Modell:",
        "label.device": "Gerät:",
        "label.language": "Sprache:",
        "label.voice": "Stimme:",
        "label.output_file": "Ausgabedatei:",
        "label.search": "Verlauf durchsuchen...",
        "label.batch_queue": "Warteschlange",
        "menu.file": "&Datei",
        "menu.tools": "&Werkzeuge",
        "menu.view": "&Ansicht",
        "menu.help": "&Hilfe",
        "warn.no_watermark": (
            "Wasserzeichen deaktiviert. Die Verantwortung für die "
            "KI-Inhaltskennzeichnung liegt beim Betreiber (EU AI Act Art. 50)."
        ),
        "warn.no_transcription": "Keine Transkription zum Speichern vorhanden.",
        "wizard.step1_title": "Schritt 1: Referenzaudio auswählen",
        "wizard.step2_title": "Schritt 2: Referenztranskription",
        "wizard.step3_title": "Schritt 3: Bestätigen",
    },
}


def set_locale(locale):
    """Set the active locale (e.g. 'en', 'de')."""
    global _current_locale
    if locale in TRANSLATIONS:
        _current_locale = locale
    else:
        logger.warning("Unknown locale '%s', falling back to 'en'", locale)
        _current_locale = "en"


def get_locale():
    """Return the current locale."""
    return _current_locale


def t(key):
    """Translate a key to the current locale. Falls back to English."""
    strings = TRANSLATIONS.get(_current_locale, TRANSLATIONS["en"])
    if key in strings:
        return strings[key]
    # Fallback to English
    return TRANSLATIONS["en"].get(key, key)


def available_locales():
    """Return list of available locale codes."""
    return list(TRANSLATIONS.keys())
