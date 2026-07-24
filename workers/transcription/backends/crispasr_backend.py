# workers/transcription/backends/crispasr_backend.py
"""CrispASR backend — unified multi-model ASR/TTS/translation via the crispasr binary.

Supports all CrispASR backends (whisper, parakeet, canary, cohere, granite,
qwen3, voxtral, voxtral4b, fastconformer-ctc, wav2vec2, moonshine,
kyutai-stt, firered-asr, omniasr, vibevoice, glm-asr, funasr, gemma4-e2b,
and more) through a single interface.

Also supports TTS (kokoro, orpheus, qwen3-tts, chatterbox, vibevoice,
indextts, voxcpm2-tts, melotts, piper, bark, dia, zonos, csm, and more)
and translation (m2m100, m2m100-wmt21, madlad, gemma4-e2b).

Synced with CrispASR 0.8.22.

The backend auto-detects from the GGUF file metadata, or can be forced
with the `crispasr_backend` kwarg.

Requires the `crispasr` binary on PATH or at CRISPASR_EXECUTABLE.
Build from https://github.com/CrispStrobe/CrispASR
"""

import logging
import os
import re
import subprocess
import threading
import time

from .base import TranscriptionBackend

# ---------------------------------------------------------------------------
# Parameter map: Python kwarg -> (CLI flag, type)
# bool  → flag only (no value), str/int/float → flag + value
# ---------------------------------------------------------------------------
PARAM_MAP = {
    # --- Sub-backend ---
    "crispasr_backend": ("--backend", str),
    "diagnostics": ("--diagnostics", bool),
    "input_file": ("--file", str),
    # --- Basic inference ---
    "threads": ("-t", int),
    "processors": ("-p", int),
    "offset_t": ("-ot", int),
    "offset_n": ("-on", int),
    "duration": ("-d", int),
    "max_context": ("-mc", int),
    "max_len": ("-ml", int),
    "split_on_punct": ("--split-on-punct", bool),
    "split_on_word": ("--split-on-word", bool),
    "best_of": ("--best-of", int),
    "beam_size": ("--beam-size", int),
    "audio_ctx": ("-ac", int),
    "word_thold": ("-wt", float),
    "entropy_thold": ("-et", float),
    "logprob_thold": ("-lpt", float),
    "no_speech_thold": ("-nth", float),
    "temperature": ("-tp", float),
    "seed": ("--seed", int),
    "temperature_inc": ("-tpi", float),
    "debug_mode": ("--debug-mode", bool),
    "prompt": ("--prompt", str),
    "carry_initial_prompt": ("--carry-initial-prompt", bool),
    "model_quant": ("--model-quant", str),
    "no_gpu": ("--no-gpu", bool),
    "flash_attn": ("--flash-attn", bool),
    "no_flash_attn": ("--no-flash-attn", bool),
    "gpu_device": ("--device", int),
    "gpu_backend": ("--gpu-backend", str),
    "n_gpu_layers": ("-ngl", int),
    "no_kv_offload": ("--no-kv-offload", bool),
    "att_context": ("--att-context", int),
    "suppress_nst": ("--suppress-nst", bool),
    "suppress_regex": ("--suppress-regex", str),
    "translate": ("--translate", bool),
    "no_fallback": ("--no-fallback", bool),
    "max_new_tokens": ("-n", int),
    "frequency_penalty": ("--frequency-penalty", float),
    "chunk_seconds": ("--chunk-seconds", int),
    "chunk_overlap": ("--chunk-overlap", float),
    "lcs_dedup": ("--lcs-dedup", str),
    "lcs_min_length": ("--lcs-min-length", int),
    "ask": ("--ask", str),
    "context": ("--context", str),
    "prefix_text": ("--prefix-text", str),
    # --- Output formats ---
    "output_txt": ("-otxt", bool),
    "output_vtt": ("-ovtt", bool),
    "output_srt": ("-osrt", bool),
    "output_words": ("-owts", bool),
    "output_csv": ("-ocsv", bool),
    "output_json": ("-oj", bool),
    "output_json_full": ("-ojf", bool),
    "output_lrc": ("-olrc", bool),
    "output_diarized_json": ("-odjson", bool),
    "output_file": ("-of", str),
    "no_prints": ("-np", bool),
    "verbose": ("--verbose", bool),
    "print_special": ("-ps", bool),
    "print_colors": ("-pc", bool),
    "print_confidence": ("--print-confidence", bool),
    "print_progress": ("-pp", bool),
    "no_timestamps": ("-nt", bool),
    "font_path": ("-fp", str),
    "log_score": ("--log-score", bool),
    "return_logits": ("--return-logits", bool),
    # --- VAD ---
    "vad": ("--vad", bool),
    "vad_model": ("-vm", str),
    "vad_threshold": ("-vt", float),
    "vad_min_speech_duration_ms": ("-vspd", int),
    "vad_min_silence_duration_ms": ("-vsd", int),
    "vad_max_speech_duration_s": ("-vmsd", float),
    "vad_speech_pad_ms": ("-vp", int),
    "vad_samples_overlap": ("-vo", float),
    "vad_stitch": ("--vad-stitch", bool),
    "vad_export": ("--vad-export", str),
    "vad_import": ("--vad-import", str),
    "vad_import_strict": ("--vad-import-strict", bool),
    "vad_export_raw": ("--vad-export-raw", str),
    # --- Diarization ---
    "diarize": ("--diarize", bool),
    "tinydiarize": ("--tinydiarize", bool),
    "diarize_method": ("--diarize-method", str),
    "diarize_embedder": ("--diarize-embedder", str),
    "diarize_cluster_threshold": ("--diarize-cluster-threshold", float),
    "diarize_max_speakers": ("--diarize-max-speakers", int),
    "diarize_speakers": ("--diarize-speakers", bool),
    "sherpa_bin": ("--sherpa-bin", str),
    "sherpa_segment_model": ("--sherpa-segment-model", str),
    "sherpa_embedding_model": ("--sherpa-embedding-model", str),
    "sherpa_num_clusters": ("--sherpa-num-clusters", int),
    # --- Language ID ---
    "detect_language": ("--detect-language", bool),
    "lid_backend": ("--lid-backend", str),
    "lid_model": ("--lid-model", str),
    "lid_on_transcript": ("--lid-on-transcript", str),
    # --- Source/target language ---
    "source_lang": ("-sl", str),
    "target_lang": ("-tl", str),
    # --- Alignment ---
    "aligner_model": ("-am", str),
    "force_aligner": ("-falign", bool),
    "no_auto_aligner": ("--no-auto-aligner", bool),
    "align_only": ("--align-only", bool),
    "text_file": ("--text-file", str),
    "align_output": ("--align-output", str),
    "align_format": ("--align-format", str),
    "align": ("--align", bool),
    "align_granularity": ("--align-granularity", str),
    # --- Punctuation ---
    "no_punctuation": ("--no-punctuation", bool),
    "punc_model": ("--punc-model", str),
    "truecase_model": ("--truecase-model", str),
    # --- Speaker ---
    "speaker_db": ("--speaker-db", str),
    "expect_speakers": ("--expect-speakers", str),
    "enroll_speaker": ("--enroll-speaker", str),
    "speaker_threshold": ("--speaker-threshold", float),
    "titanet_model": ("--titanet-model", str),
    "speaker_db_consent": ("--speaker-db-consent", bool),
    # --- Grammar ---
    "grammar": ("--grammar", str),
    "grammar_rule": ("--grammar-rule", str),
    "grammar_penalty": ("--grammar-penalty", float),
    # --- DTW ---
    "dtw": ("--dtw", str),
    # --- Alternatives ---
    "alt": ("--alt", bool),
    "alt_n": ("--alt-n", int),
    # --- Streaming ---
    "stream": ("--stream", bool),
    "mic": ("--mic", bool),
    "live": ("--live", bool),
    "monitor": ("--monitor", bool),
    "stream_step": ("--stream-step", int),
    "stream_length": ("--stream-length", int),
    "stream_keep": ("--stream-keep", int),
    "stream_json": ("--stream-json", bool),
    "stream_final_on_silence_ms": ("--stream-final-on-silence-ms", int),
    "stream_vad_merge_gap_ms": ("--stream-vad-merge-gap-ms", int),
    "stream_partial_decode_ms": ("--stream-partial-decode-ms", int),
    "stream_punc": ("--stream-punc", str),
    "stream_final_mode": ("--stream-final-mode", str),
    "stream_utterance_max_sec": ("--stream-utterance-max-sec", float),
    # --- Server ---
    "server": ("--server", bool),
    "host": ("--host", str),
    "port": ("--port", int),
    "ws_port": ("--ws-port", int),
    "api_keys": ("--api-keys", str),
    "cors_origin": ("--cors-origin", str),
    "no_warmup": ("--no-warmup", bool),
    "wyoming_port": ("--wyoming-port", int),
    # --- Speech-to-speech ---
    "s2s": ("--s2s", bool),
    "s2s_output": ("--s2s-output", str),
    # --- TTS ---
    "tts_text": ("--tts", str),
    "tts_output": ("--tts-output", str),
    "tts_voice": ("--voice", str),
    "tts_ref_text": ("--ref-text", str),
    "tts_ref_asr": ("--ref-asr", str),
    "tts_instruct": ("--instruct", str),
    "tts_codec_model": ("--codec-model", str),
    "tts_codec_quant": ("--codec-quant", str),
    "tts_steps": ("--tts-steps", int),
    "tts_trim_silence": ("--tts-trim-silence", bool),
    "tts_stream": ("--tts-stream", bool),
    "tts_cfg_scale": ("--tts-cfg-scale", float),
    "tts_speed": ("--tts-speed", float),
    "tts_max_input_chars": ("--tts-max-input-chars", int),
    "tts_play": ("--tts-play", bool),
    "tts_play_device": ("--tts-play-device", int),
    "voice_dir": ("--voice-dir", str),
    "g2p_dict": ("--g2p-dict", str),
    "make_ref": ("--make-ref", bool),
    "make_ref_output": ("--make-ref-output", str),
    "make_ref_encoder": ("--make-ref-encoder", str),
    "make_ref_aligner": ("--make-ref-aligner", str),
    # --- Provenance / EU AI Act (voice cloning, watermarking, C2PA) ---
    "i_have_rights": ("--i-have-rights", bool),
    "accept_license": ("--accept-license", str),
    "no_spoken_disclaimer": ("--no-spoken-disclaimer", bool),
    "no_watermark": ("--no-watermark", bool),
    "no_c2pa": ("--no-c2pa", bool),
    "accept_marking_responsibility": ("--accept-marking-responsibility", bool),
    "watermark_model": ("--watermark-model", str),
    "detect_watermark": ("--detect-watermark", str),
    "c2pa_cert": ("--c2pa-cert", str),
    "c2pa_key": ("--c2pa-key", str),
    # --- Translation (text-to-text) ---
    "text_input": ("--text", str),
    "tr_source_lang": ("--tr-sl", str),
    "tr_target_lang": ("--tr-tl", str),
    "translate_max_tokens": ("--translate-max-tokens", int),
    # --- Model download ---
    "auto_download": ("--auto-download", bool),
    "hf_repo": ("--hf-repo", str),
    "hf_file": ("--hf-file", str),
    "cache_dir": ("--cache-dir", str),
    "dry_run_resolve": ("--dry-run-resolve", bool),
    "dry_run_ignore_cache": ("--dry-run-ignore-cache", bool),
    # --- Misc ---
    "flush_after": ("--flush-after", int),
    "ov_e_device": ("--ov-e-device", str),
    "firered_vad_debug": ("--firered-vad-debug", bool),
    # parakeet-decoder takes a value (ctc|tdt|maes), not a bare flag
    "parakeet_decoder": ("--parakeet-decoder", str),
    # --- Hotwords / contextual biasing ---
    "hotwords": ("--hotwords", str),
    "hotwords_file": ("--hotwords-file", str),
    "hotwords_boost": ("--hotwords-boost", float),
    # --- Chat (server only) ---
    "chat_model": ("--chat-model", str),
    "chat_ctx": ("--chat-ctx", int),
    "chat_gpu_layers": ("--chat-gpu-layers", int),
    # --- Audio analysis / source separation ---
    "separate": ("--separate", bool),
    "stems": ("--stems", str),
    "sep_output_dir": ("--sep-output-dir", str),
    "pitch": ("--pitch", bool),
    "pitch_format": ("--pitch-format", str),
    "pitch_hop_ms": ("--pitch-hop-ms", float),
    "piano": ("--piano", bool),
    "piano_format": ("--piano-format", str),
    "chords": ("--chords", bool),
    "chords_format": ("--chords-format", str),
    "tab": ("--tab", bool),
    "tab_format": ("--tab-format", str),
    "beats": ("--beats", bool),
    "beats_format": ("--beats-format", str),
}

# Timestamp regex for output parsing
_TS_RE = re.compile(r"\[(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]\s*(.*)")


class CrispasrBackend(TranscriptionBackend):
    """CrispASR backend — calls the crispasr binary for any supported model.

    model_id should be a path to a GGUF model file, or "auto" / "auto:<quant>"
    for auto-download from the model registry.

    All CrispASR CLI parameters can be passed as kwargs using their Python
    names (underscores instead of hyphens). See PARAM_MAP for the full list.

    Common kwargs:
        crispasr_backend: str — force a specific backend (e.g. "parakeet")
        word_timestamps: bool — request word-level timestamps
        vad: bool — enable Silero VAD for long audio
        split_on_punct: bool — split subtitles at sentence boundaries
        temperature: float — sampling temperature (0 = greedy)
        best_of: int — best-of-N candidates with temperature > 0
        auto_download: bool — auto-download model without prompting
        seed: int — RNG seed for reproducibility
        beam_size: int — beam search width
        diarize: bool — enable speaker diarization
        diarize_method: str — diarization method (energy/xcorr/vad-turns/pyannote/sherpa/ecapa)
        punc_model: str — punctuation restoration model
        aligner_model: str — CTC aligner for word timestamps
    """

    def __init__(self, model_id, device, language=None, word_timestamps=False, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.extra_kwargs = kwargs
        self.temp_files = []

    def _build_base_cmd(self):
        """Build the base command with the crispasr binary and model."""
        from utils.crispasr_utils import find_crispasr

        exe = find_crispasr()
        if not exe:
            raise FileNotFoundError(
                "crispasr binary not found. Set CRISPASR_EXECUTABLE or "
                "install from https://github.com/CrispStrobe/CrispASR"
            )

        cmd = [exe]

        # Model — support "auto" and "auto:quant" for auto-download
        if self.model_id:
            model_str = str(self.model_id)
            if model_str.startswith("auto"):
                cmd.extend(["-m", model_str])
                if "auto_download" not in self.extra_kwargs:
                    cmd.append("--auto-download")
            elif os.path.isfile(model_str):
                cmd.extend(["-m", model_str])
            else:
                # Could be a registry name or HF repo
                cmd.extend(["-m", model_str])

        return cmd, exe

    def _append_params(self, cmd):
        """Append all configured parameters from kwargs to the command."""
        for kwarg_name, (flag, param_type) in PARAM_MAP.items():
            value = self.extra_kwargs.get(kwarg_name)
            if value is None:
                continue
            if param_type is bool:
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

    def _run_process(self, cmd):
        """Run the crispasr binary and yield parsed output lines."""
        logging.info(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def collect_stderr():
            for line in iter(process.stderr.readline, ""):
                line = line.strip()
                if line:
                    logging.info(f"crispasr: {line}")

        stderr_thread = threading.Thread(target=collect_stderr, daemon=True)
        stderr_thread.start()

        output_lines = []
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            m = _TS_RE.match(line)
            if m:
                start = self._parse_ts(m.group(1))
                end = self._parse_ts(m.group(2))
                text = m.group(3).strip()
                if text:
                    output_lines.append((start, end, text))
            else:
                if line:
                    output_lines.append((0.0, 0.0, line))

        rc = process.wait()
        stderr_thread.join(timeout=5)

        if rc != 0:
            logging.error(f"crispasr failed with code {rc}")
            raise RuntimeError(f"crispasr exited with code {rc}")

        return output_lines

    def preprocess_audio(self, audio_path):
        """Convert to WAV if needed (crispasr handles WAV/MP3/FLAC/OGG)."""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".ogg"):
            return audio_path
        from utils.audio_utils import convert_audio_to_wav

        wav_path = convert_audio_to_wav(audio_path)
        if wav_path != audio_path:
            self.temp_files.append(wav_path)
        return wav_path

    def transcribe(self, audio_path):
        """Transcribe using the crispasr binary."""
        logging.info("=== Starting CrispASR pipeline ===")

        cmd, exe = self._build_base_cmd()
        logging.info(f"Using crispasr: {exe}")

        cmd.extend(["-f", audio_path])

        # Default: use available CPU threads (capped at 8) and suppress progress
        if "threads" not in self.extra_kwargs:
            cmd.extend(["-t", str(min(os.cpu_count() or 4, 8))])
        if "no_prints" not in self.extra_kwargs:
            cmd.append("-np")

        # Language
        if self.language:
            cmd.extend(["-l", self.language])

        # Word timestamps via max_len=1
        if self.word_timestamps and "max_len" not in self.extra_kwargs:
            cmd.extend(["-ml", "1"])

        # Append all extra params
        self._append_params(cmd)

        t0 = time.time()
        results = self._run_process(cmd)
        elapsed = time.time() - t0

        # Compute metrics
        word_count = sum(len(r[2].split()) for r in results)
        try:
            audio_len = self._get_audio_duration(audio_path)
        except Exception:
            audio_len = 0
        if audio_len > 0 and elapsed > 0:
            from utils.crispasr_utils import compute_metrics

            metrics = compute_metrics(audio_len, elapsed, word_count)
            logging.info(
                "Metrics: RTF=%.1fx, %.0f WPS, %.1fs audio in %.1fs",
                metrics["rtf"],
                metrics["wps"],
                metrics["audio_s"],
                metrics["wall_s"],
            )

        for result in results:
            yield result

    @staticmethod
    def _get_audio_duration(audio_path):
        """Get audio duration in seconds (best-effort)."""
        try:
            import wave

            with wave.open(audio_path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            pass
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception:
            return 0

    def synthesize(self, text, output_path="tts_output.wav"):
        """Synthesize text to audio using CrispASR's TTS capabilities.

        Returns the path to the output audio file.
        """
        logging.info("=== Starting CrispASR TTS ===")

        cmd, exe = self._build_base_cmd()
        logging.info(f"Using crispasr: {exe}")

        cmd.extend(["--tts", text, "--tts-output", output_path])

        if "threads" not in self.extra_kwargs:
            cmd.extend(["-t", str(min(os.cpu_count() or 4, 8))])

        self._append_params(cmd)

        logging.info(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()

        if stderr:
            for line in stderr.strip().splitlines():
                logging.info(f"crispasr: {line}")

        if process.returncode != 0:
            raise RuntimeError(f"CrispASR TTS failed with code {process.returncode}: {stderr}")

        if os.path.isfile(output_path):
            logging.info(f"TTS output saved to: {output_path}")
            return output_path

        raise FileNotFoundError(f"TTS output not found at: {output_path}")

    def translate_text(self, text, source_lang="en", target_lang="de"):
        """Translate text using CrispASR's translation backends (m2m100, etc.).

        Returns the translated text.
        """
        logging.info("=== Starting CrispASR Translation ===")

        cmd, exe = self._build_base_cmd()
        logging.info(f"Using crispasr: {exe}")

        cmd.extend(["--text", text])
        if source_lang:
            cmd.extend(["--tr-sl", source_lang])
        if target_lang:
            cmd.extend(["--tr-tl", target_lang])

        if "threads" not in self.extra_kwargs:
            cmd.extend(["-t", str(min(os.cpu_count() or 4, 8))])

        # Add translation-specific params but skip text_input/tr_source_lang/tr_target_lang
        # since we already added them above
        skip = {"text_input", "tr_source_lang", "tr_target_lang"}
        for kwarg_name, (flag, param_type) in PARAM_MAP.items():
            if kwarg_name in skip:
                continue
            value = self.extra_kwargs.get(kwarg_name)
            if value is None:
                continue
            if param_type is bool:
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        logging.info(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()

        if stderr:
            for line in stderr.strip().splitlines():
                logging.info(f"crispasr: {line}")

        if process.returncode != 0:
            raise RuntimeError(
                f"CrispASR translation failed with code {process.returncode}: {stderr}"
            )

        return stdout.strip()

    def start_server(self, host="127.0.0.1", port=8080):
        """Start CrispASR in server mode (blocking).

        Returns the subprocess.Popen object for the caller to manage.
        """
        logging.info("=== Starting CrispASR Server ===")

        cmd, exe = self._build_base_cmd()
        cmd.extend(["--server", "--host", host, "--port", str(port)])

        if "threads" not in self.extra_kwargs:
            cmd.extend(["-t", str(min(os.cpu_count() or 4, 8))])

        self._append_params(cmd)

        logging.info(f"Starting server: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    @staticmethod
    def _parse_ts(ts_str):
        """Parse HH:MM:SS.mmm to seconds."""
        parts = ts_str.split(":")
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

    def cleanup(self):
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {f}: {e}")
