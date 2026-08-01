# workers/tts_thread.py
"""QThread for TTS synthesis (non-blocking GUI operation)."""

import logging
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from utils.i18n import t

#: Display names for the provenance layers, in the order they are applied.
_LAYER_NAMES = (
    ("spoken", "spoken disclosure"),
    ("watermark", "watermark"),
    ("marker", "AI marker"),
    ("c2pa", "C2PA"),
)


def _describe_marking(marking):
    """Render an ``apply_provenance()`` result as a user-facing status line."""
    if marking.get("opted_out"):
        return t("warn.marking_opted_out")
    layers = [label for key, label in _LAYER_NAMES if marking.get(key)]
    if not layers:
        if marking.get("unsupported_format"):
            return t("warn.marking_unsupported_format")
        return t("warn.marking_failed")
    return t("status.marked").format(layers=" + ".join(layers))


#: Provenance opt-outs that require the responsibility attestation, mirroring
#: the CLI's ``_MARKING_OPT_OUTS`` and the CrispASR binary's own rule.
_OPT_OUT_KEYS = ("no_watermark", "no_c2pa", "no_spoken_disclaimer")


def _refuse_unattested_opt_out(args):
    """Return an error string if provenance is reduced without attestation."""
    if args.get("accept_marking_responsibility"):
        return None
    if not any(args.get(key) for key in _OPT_OUT_KEYS):
        return None
    return t("error.marking_opt_out_needs_attestation")


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

            refusal = _refuse_unattested_opt_out(self.args)
            if refusal:
                self.error_signal.emit(refusal)
                return

            self.progress_signal.emit(t("status.tts_initializing").format(backend=backend_name))

            # Provenance / EU AI Act kwargs apply to *both* branches: the
            # CrispASR binary consumes them as flags, the Python-native
            # backends via TTSBackend.require_clone_consent/apply_provenance.
            provenance = {}
            for key in (
                "i_have_rights",
                "no_spoken_disclaimer",
                "no_watermark",
                "no_c2pa",
                "accept_marking_responsibility",
                "c2pa_cert",
                "c2pa_key",
            ):
                if self.args.get(key):
                    provenance[key] = self.args[key]

            if self.args.get("no_watermark"):
                self.progress_signal.emit(t("warn.no_watermark"))
                logging.warning(t("warn.no_watermark"))

            if backend_name.startswith("crispasr"):
                from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

                kwargs = dict(provenance)
                if ":" in backend_name:
                    kwargs["crispasr_backend"] = backend_name.split(":", 1)[1]
                if self.args.get("reference_audio"):
                    kwargs["voice"] = self.args["reference_audio"]
                if self.args.get("auto_download", True):
                    kwargs["auto_download"] = True
                if self.args.get("ref_text"):
                    kwargs["ref_text"] = self.args["ref_text"]
                if self.args.get("g2p_dict"):
                    kwargs["g2p_dict"] = self.args["g2p_dict"]

                backend = CrispasrTTSBackend(
                    model_id=model_id, device=device, language=language, **kwargs
                )
            else:
                from workers.tts.backends import get_tts_backend

                kwargs = dict(provenance)
                if voice:
                    kwargs["voice"] = voice
                if self.args.get("reference_audio"):
                    kwargs["reference_audio"] = self.args["reference_audio"]

                backend = get_tts_backend(
                    backend_name,
                    model_id=model_id,
                    device=device,
                    language=language,
                    **kwargs,
                )

            if self._stopped:
                return

            self.progress_signal.emit(t("status.tts_synthesizing").format(backend=backend_name))
            result = backend.synthesize(text, output_path, voice=voice)

            # EU AI Act Art. 50(2): mark synthetic audio as machine-readable.
            # CrispASR backends verify the binary's own marking instead.
            # `language` drives the spoken disclosure so a German UI discloses
            # in German — it was previously dropped, defaulting every
            # disclosure to the process locale.
            marking = backend.apply_provenance(result, model=model_id, voice=voice, locale=language)
            self.progress_signal.emit(_describe_marking(marking))

            backend.cleanup()

            if self._stopped:
                return

            self.progress_signal.emit(t("status.audio_saved").format(path=result))
            self.finished_signal.emit(result)

        except PermissionError as e:
            # Voice-cloning consent gate — a refusal, not a crash.
            logging.warning(f"TTS refused: {e}")
            self.error_signal.emit(str(e))
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
                backend_name,
                model_id=model_id,
                **kwargs,
            )

            result = backend.translate(text, source_lang, target_lang)
            backend.cleanup()

            self.progress_signal.emit("Translation complete.")
            self.result_signal.emit(result)

        except Exception as e:
            logging.error(f"Translation error: {e}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))
