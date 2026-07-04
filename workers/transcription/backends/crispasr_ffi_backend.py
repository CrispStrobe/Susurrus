# workers/transcription/backends/crispasr_ffi_backend.py
"""CrispASR FFI backend — direct Python bindings via ctypes.

Uses ``crispasr.Session`` to call libcrispasr in-process rather than
spawning a subprocess. This gives:

- Zero IPC overhead (no process spawn, no stdout parsing)
- Persistent model sessions (load once, transcribe many)
- Native structured output (SessionSegment/SessionWord objects)
- In-process TTS (PCM array, no temp files)
- In-process translation

Falls back to the subprocess backend if libcrispasr is not loadable.

Requires the crispasr Python package::

    pip install crispasr
    # or: sys.path includes whisper.cpp/python/
"""

import logging
import os
import time
import wave

from .base import TranscriptionBackend

logger = logging.getLogger(__name__)


def _ffi_available():
    """Check if the crispasr FFI bindings are importable."""
    try:
        import importlib.util

        return importlib.util.find_spec("crispasr") is not None
    except (ImportError, OSError):
        return False


class CrispasrFFIBackend(TranscriptionBackend):
    """CrispASR FFI backend — in-process inference via libcrispasr.

    model_id should be a path to a GGUF model file.

    The session is opened once on the first call and reused for
    subsequent transcriptions (persistent model).

    Kwargs:
        crispasr_backend: str — force a specific backend (e.g. "parakeet")
        vad: bool — enable Silero VAD
        vad_model: str — path to VAD model GGUF
        vad_threshold: float — VAD threshold (0.0-1.0)
        temperature: float — sampling temperature
        seed: int — RNG seed
        beam_size: int — beam search width
        best_of: int — best-of-N
        translate: bool — translate to English (whisper only)
        source_lang: str — source language hint
        target_lang: str — target language for translation
        punc_model: str — punctuation restoration model path
        max_new_tokens: int — max generated tokens
        frequency_penalty: float — frequency penalty
        grammar: str — GBNF grammar text
        grammar_rule: str — grammar root rule
        grammar_penalty: float — grammar penalty weight
        ask: str — free-form prompt for instruct backends
        n_threads: int — compute threads (default: min(cpu_count, 8))
    """

    def __init__(self, model_id, device, language=None, word_timestamps=False, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.extra_kwargs = kwargs
        self._session = None
        self._punc_model = None
        self.temp_files = []

    def _ensure_session(self):
        """Open a CrispASR session if not already open.

        Raises RuntimeError with a helpful message if libcrispasr is
        not available — callers can catch this and fall back to the
        subprocess-based backend.
        """
        if self._session is not None:
            return

        try:
            from crispasr import Session
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f"CrispASR FFI not available: {e}. "
                "Use 'crispasr' (subprocess) backend instead, or build "
                "libcrispasr as a shared library (BUILD_SHARED_LIBS=ON)."
            ) from e

        n_threads = self.extra_kwargs.get("n_threads", min(os.cpu_count() or 4, 8))
        backend = self.extra_kwargs.get("crispasr_backend")

        logger.info(
            "Opening CrispASR FFI session: model=%s, backend=%s, threads=%d",
            self.model_id,
            backend,
            n_threads,
        )

        self._session = Session(
            self.model_id,
            n_threads=n_threads,
            backend=backend,
        )
        logger.info("Session opened — detected backend: %s", self._session.backend)

        # Apply sticky session state from kwargs
        self._apply_session_state()

    def _apply_session_state(self):
        """Apply all configured session parameters."""
        s = self._session
        kw = self.extra_kwargs

        # Language
        if kw.get("source_lang"):
            s.set_source_language(kw["source_lang"])
        if kw.get("target_lang"):
            s.set_target_language(kw["target_lang"])

        # Translate (whisper EN-only)
        if kw.get("translate"):
            s.set_translate(True)

        # Temperature
        temp = kw.get("temperature")
        if temp is not None:
            seed = kw.get("seed", 0)
            s.set_temperature(float(temp), int(seed))

        # Beam search / best-of
        if kw.get("beam_size"):
            s.set_beam_size(int(kw["beam_size"]))
        if kw.get("best_of"):
            s.set_best_of(int(kw["best_of"]))

        # Token limits
        if kw.get("max_new_tokens"):
            s.set_max_new_tokens(int(kw["max_new_tokens"]))
        if kw.get("frequency_penalty"):
            s.set_frequency_penalty(float(kw["frequency_penalty"]))

        # Grammar
        if kw.get("grammar"):
            rule = kw.get("grammar_rule", "root")
            penalty = kw.get("grammar_penalty", 100.0)
            s.set_grammar_text(kw["grammar"], rule, float(penalty))

        # Punctuation
        if kw.get("no_punctuation"):
            s.set_punctuation(False)

        # Ask / prompt
        if kw.get("ask"):
            s.set_ask(kw["ask"])

        # Hotwords / contextual biasing (CrispASR 0.7.x). Guarded with hasattr
        # so older libcrispasr builds that lack the setter degrade gracefully.
        if kw.get("hotwords") and hasattr(s, "set_hotwords"):
            boost = float(kw.get("hotwords_boost", 2.0))
            s.set_hotwords(kw["hotwords"], boost)

        # G2P dictionary for TTS phonemization (CrispASR 0.7.x)
        if kw.get("g2p_dict") and hasattr(s, "set_g2p_dict"):
            s.set_g2p_dict(kw["g2p_dict"])

        # TTS-specific
        if kw.get("tts_voice"):
            ref_text = kw.get("tts_ref_text")
            s.set_voice(kw["tts_voice"], ref_text=ref_text)
        if kw.get("tts_codec_model"):
            s.set_codec_path(kw["tts_codec_model"])
        if kw.get("tts_instruct"):
            s.set_instruct(kw["tts_instruct"])
        if kw.get("tts_steps"):
            s.set_tts_steps(int(kw["tts_steps"]))

        # TTS sampling
        if kw.get("top_p"):
            s.set_top_p(float(kw["top_p"]))
        if kw.get("min_p"):
            s.set_min_p(float(kw["min_p"]))
        if kw.get("repetition_penalty"):
            s.set_repetition_penalty(float(kw["repetition_penalty"]))
        if kw.get("cfg_weight"):
            s.set_cfg_weight(float(kw["cfg_weight"]))
        if kw.get("exaggeration"):
            s.set_exaggeration(float(kw["exaggeration"]))
        if kw.get("length_scale"):
            s.set_length_scale(float(kw["length_scale"]))

        # Top-K / multinomial sampling (CrispASR 0.8.x)
        if kw.get("top_k") and hasattr(s, "set_top_k"):
            s.set_top_k(int(kw["top_k"]))
        if kw.get("do_sample") is not None and hasattr(s, "set_do_sample"):
            s.set_do_sample(bool(kw["do_sample"]))

        # TTS candidate count and noise temperature (CrispASR 0.8.x, TADA)
        if kw.get("tts_num_candidates") and hasattr(s, "set_tts_num_candidates"):
            s.set_tts_num_candidates(int(kw["tts_num_candidates"]))
        if kw.get("tts_noise_temp") is not None and hasattr(s, "set_tts_noise_temp"):
            s.set_tts_noise_temp(float(kw["tts_noise_temp"]))

        # Fallback thresholds (whisper)
        if kw.get("entropy_thold") is not None:
            s.set_fallback_thresholds(
                entropy_threshold=float(kw.get("entropy_thold", 2.4)),
                logprob_threshold=float(kw.get("logprob_thold", -1.0)),
                no_speech_threshold=float(kw.get("no_speech_thold", 0.6)),
                temperature_inc=float(kw.get("temperature_inc", 0.2)),
            )

        # Alt token candidates
        if kw.get("alt_n"):
            s.set_alt_n(int(kw["alt_n"]))

    def preprocess_audio(self, audio_path):
        """Convert to WAV if needed and load as float32 PCM."""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg"):
            from utils.audio_utils import convert_audio_to_wav

            wav_path = convert_audio_to_wav(audio_path)
            if wav_path != audio_path:
                self.temp_files.append(wav_path)
            return wav_path
        return audio_path

    def _load_audio(self, audio_path):
        """Load audio file as float32 PCM at 16 kHz."""
        import numpy as np

        try:
            import soundfile as sf

            pcm, sr = sf.read(audio_path, dtype="float32")
            if len(pcm.shape) > 1:
                pcm = pcm.mean(axis=1)
            if sr != 16000:
                ratio = 16000 / sr
                new_len = int(len(pcm) * ratio)
                indices = np.linspace(0, len(pcm) - 1, new_len)
                pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
            return pcm
        except ImportError:
            pass

        # Fallback: wave module (WAV only)
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sample_width == 2:
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)

        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

        return pcm

    def transcribe(self, audio_path):
        """Transcribe using CrispASR FFI (in-process)."""
        logger.info("=== Starting CrispASR FFI pipeline ===")
        self._ensure_session()

        pcm = self._load_audio(audio_path)
        audio_seconds = len(pcm) / 16000.0
        logger.info("Audio loaded: %.1fs (%d samples)", audio_seconds, len(pcm))

        t0 = time.time()

        # Choose VAD or direct transcription
        vad_model = self.extra_kwargs.get("vad_model")
        use_vad = self.extra_kwargs.get("vad", False)

        if use_vad and vad_model:
            segments = self._session.transcribe_vad(
                pcm,
                vad_model,
                threshold=float(self.extra_kwargs.get("vad_threshold", 0.5)),
                min_speech_duration_ms=int(
                    self.extra_kwargs.get("vad_min_speech_duration_ms", 250)
                ),
                min_silence_duration_ms=int(
                    self.extra_kwargs.get("vad_min_silence_duration_ms", 100)
                ),
                speech_pad_ms=int(self.extra_kwargs.get("vad_speech_pad_ms", 30)),
                chunk_seconds=int(self.extra_kwargs.get("chunk_seconds", 30)),
                n_threads=int(self.extra_kwargs.get("n_threads", min(os.cpu_count() or 4, 8))),
                language=self.language,
            )
        elif use_vad:
            # VAD enabled but no model path — use default cache path
            from crispasr import cache_ensure_file, registry_lookup

            try:
                entry = registry_lookup("silero-vad")
                if entry:
                    vad_path = cache_ensure_file(entry.filename, entry.url, quiet=True)
                    segments = self._session.transcribe_vad(
                        pcm,
                        vad_path,
                        language=self.language,
                    )
                else:
                    segments = self._session.transcribe(pcm, language=self.language)
            except Exception:
                segments = self._session.transcribe(pcm, language=self.language)
        else:
            segments = self._session.transcribe(pcm, language=self.language)

        elapsed = time.time() - t0
        rtf = audio_seconds / elapsed if elapsed > 0 else 0
        logger.info(
            "Transcription done: %d segments in %.1fs (RTF=%.1fx)",
            len(segments),
            elapsed,
            rtf,
        )

        # Punctuation post-processing
        punc_path = self.extra_kwargs.get("punc_model")
        if punc_path:
            segments = self._apply_punctuation(segments, punc_path)

        # Yield results in the standard (start, end, text) format
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            if self.word_timestamps and seg.words:
                for word in seg.words:
                    wtext = word.text.strip()
                    if wtext:
                        yield (word.start, word.end, wtext)
            else:
                yield (seg.start, seg.end, text)

    def _apply_punctuation(self, segments, punc_path):
        """Apply punctuation restoration to segments."""
        if self._punc_model is None:
            try:
                from crispasr._binding import PuncModel

                self._punc_model = PuncModel(punc_path)
            except Exception as e:
                logger.warning("Failed to load punctuation model: %s", e)
                return segments

        for seg in segments:
            seg.text = self._punc_model.process(seg.text)
        return segments

    def synthesize(self, text, output_path="tts_output.wav"):
        """Synthesize text to audio via FFI.

        Returns the path to the output audio file.
        """
        import numpy as np

        logger.info("=== Starting CrispASR FFI TTS ===")
        self._ensure_session()

        pcm = self._session.synthesize(text)

        if pcm is None or len(pcm) == 0:
            error_msg = self.last_synth_error()
            detail = f": {error_msg}" if error_msg else ""
            raise RuntimeError(f"CrispASR FFI TTS returned empty audio{detail}")

        # Determine sample rate (backend-dependent)
        backend = self._session.backend
        if backend == "voxcpm2-tts":
            sr = 48000
        else:
            sr = 24000

        # Write WAV
        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_int16.tobytes())

        logger.info("TTS output: %s (%d samples, %d Hz)", output_path, len(pcm), sr)
        return output_path

    def speech_to_speech(self, audio_path, output_path="s2s_output.wav"):
        """Speech-to-speech via FFI (audio in -> audio out).

        Supported by backends with CAP_S2S (e.g. lfm2-audio, mini-omni2).
        Returns (output_path, transcript_or_none).
        """
        import numpy as np

        logger.info("=== Starting CrispASR FFI S2S ===")
        self._ensure_session()

        pcm_in = self._load_audio(audio_path)

        if not hasattr(self._session, "speech_to_speech"):
            raise RuntimeError(
                "speech_to_speech() not available — update the crispasr "
                "Python package to >= 0.8.0"
            )

        result = self._session.speech_to_speech(pcm_in)

        # Result is (pcm_out, transcript) or just pcm_out depending on binding version
        if isinstance(result, tuple):
            pcm_out, transcript = result
        else:
            pcm_out = result
            transcript = None

        if pcm_out is None or len(pcm_out) == 0:
            error_msg = self.last_synth_error()
            detail = f": {error_msg}" if error_msg else ""
            raise RuntimeError(f"CrispASR FFI S2S returned empty audio{detail}")

        # Write output WAV (S2S backends typically output 24 kHz)
        sr = 24000
        pcm_int16 = (pcm_out * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_int16.tobytes())

        logger.info("S2S output: %s (%d samples, %d Hz)", output_path, len(pcm_out), sr)
        return output_path, transcript

    def last_synth_error(self):
        """Return the last TTS/S2S synthesis error message, or None.

        Requires crispasr >= 0.8.0 with last_synth_error() binding.
        """
        if self._session is None:
            return None
        if hasattr(self._session, "last_synth_error"):
            return self._session.last_synth_error()
        return None

    def translate_text(self, text, source_lang="en", target_lang="de"):
        """Translate text via FFI."""
        logger.info("=== Starting CrispASR FFI Translation ===")
        self._ensure_session()
        max_tokens = self.extra_kwargs.get("translate_max_tokens", 512)
        return self._session.translate_text(text, source_lang, target_lang, max_tokens=max_tokens)

    def detect_language(self, audio_path):
        """Detect the language of an audio file.

        Returns (language_code, confidence) tuple.
        """
        self._ensure_session()
        pcm = self._load_audio(audio_path)

        from crispasr import LidMethod, detect_language_pcm

        lid_backend = self.extra_kwargs.get("lid_backend", "whisper")
        method_map = {
            "whisper": LidMethod.WHISPER,
            "silero": LidMethod.SILERO,
            "firered": LidMethod.FIRERED,
            "ecapa": LidMethod.ECAPA,
        }
        method = method_map.get(lid_backend, LidMethod.WHISPER)
        lid_model = self.extra_kwargs.get("lid_model")

        result = detect_language_pcm(pcm, 16000, method=method, model_path=lid_model)
        return (result.lang, result.confidence)

    @staticmethod
    def available_backends():
        """List backends the loaded libcrispasr was built with."""
        try:
            from crispasr import Session

            return Session.available_backends()
        except (ImportError, OSError):
            return []

    def cleanup(self):
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._punc_model is not None:
            self._punc_model.close()
            self._punc_model = None
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logger.warning("Failed to remove temp file %s: %s", f, e)
