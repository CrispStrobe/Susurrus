# workers/tts_thread.py
"""QThread for TTS synthesis (non-blocking GUI operation)."""

import logging
import traceback

from PyQt6.QtCore import QThread, pyqtSignal


class TTSThread(QThread):
    """Run TTS synthesis in a background thread.

    Signals:
        progress_signal(str): Status messages.
        error_signal(str): Error messages.
        finished_signal(str): Path to the output audio file on success.
    """

    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            backend_name = self.args["tts_backend"]
            text = self.args["text"]
            output_path = self.args.get("output_path", "tts_output.wav")
            voice = self.args.get("voice")
            model_id = self.args.get("model_id", "auto")
            device = self.args.get("device", "cpu")
            language = self.args.get("language")

            self.progress_signal.emit(f"Initializing TTS backend: {backend_name}")

            if backend_name.startswith("crispasr"):
                from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

                kwargs = {}
                if ":" in backend_name:
                    kwargs["crispasr_backend"] = backend_name.split(":", 1)[1]
                if self.args.get("reference_audio"):
                    kwargs["voice"] = self.args["reference_audio"]
                if self.args.get("auto_download", True):
                    kwargs["auto_download"] = True

                backend = CrispasrTTSBackend(
                    model_id=model_id, device=device, language=language, **kwargs
                )
            else:
                from workers.tts.backends import get_tts_backend

                kwargs = {}
                if voice:
                    kwargs["voice"] = voice

                backend = get_tts_backend(
                    backend_name, model_id=model_id, device=device,
                    language=language, **kwargs,
                )

            if self._stopped:
                return

            self.progress_signal.emit(f"Synthesizing with {backend_name}...")
            result = backend.synthesize(text, output_path, voice=voice)
            backend.cleanup()

            if self._stopped:
                return

            self.progress_signal.emit(f"Audio saved to: {result}")
            self.finished_signal.emit(result)

        except Exception as e:
            logging.error(f"TTS error: {e}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))


class TranslationThread(QThread):
    """Run translation in a background thread.

    Signals:
        progress_signal(str): Status messages.
        error_signal(str): Error messages.
        result_signal(str): Translated text.
    """

    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args

    def run(self):
        try:
            backend_name = self.args["backend"]
            text = self.args["text"]
            source_lang = self.args.get("source_lang", "en")
            target_lang = self.args.get("target_lang", "de")
            model_id = self.args.get("model_id", "auto")

            self.progress_signal.emit(f"Translating with {backend_name}...")

            from workers.translation.backends import get_translation_backend

            kwargs = {"auto_download": True}
            backend = get_translation_backend(
                backend_name, model_id=model_id, **kwargs,
            )

            result = backend.translate(text, source_lang, target_lang)
            backend.cleanup()

            self.progress_signal.emit("Translation complete.")
            self.result_signal.emit(result)

        except Exception as e:
            logging.error(f"Translation error: {e}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))
