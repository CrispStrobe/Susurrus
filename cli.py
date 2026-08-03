#!/usr/bin/env python3
"""Susurrus CLI — headless transcription, TTS, and translation.

Usage:
    # Transcription
    python cli.py --file audio.wav
    python cli.py --backend crispasr:parakeet --model auto --file audio.wav
    python cli.py --backend faster-sequenced --model large-v3 --file audio.wav

    # TTS
    python cli.py --mode tts --tts-backend edge-tts --text "Hello world"
    python cli.py --mode tts --tts-backend crispasr:kokoro --model auto --text "Hallo"

    # Translation
    python cli.py --mode translate --backend crispasr:m2m100 --model auto \\
                  --text "Hello world" --source-lang en --target-lang de

    # Streaming
    python cli.py --mode stream --backend crispasr --model auto --mic

    # Server
    python cli.py --mode server --backend crispasr --model auto --port 8080

    # List backends
    python cli.py --list-backends

Transcription backends: mlx-whisper, faster-batched, faster-sequenced,
    transformers, whisper.cpp, ctranslate2, whisper-jax, insanely-fast-whisper,
    openai whisper, voxtral-local, voxtral-api, crispasr, crispasr:<sub-backend>

TTS backends: crispasr:kokoro, crispasr:orpheus, crispasr:qwen3-tts,
    crispasr:chatterbox, crispasr:vibevoice, crispasr:melotts, crispasr:piper,
    crispasr:bark, crispasr:dia, crispasr:zonos, crispasr:csm (and more),
    edge-tts, piper, kokoro-onnx, chatterbox, speecht5

Translation backends: crispasr:m2m100, crispasr:m2m100-wmt21, crispasr:madlad,
    crispasr:gemma4-e2b
"""

import argparse
import logging
import os
import sys

from utils.provenance import ProvenanceError

sys.path.insert(0, os.path.dirname(__file__))


def get_backend_class(name):
    """Import backend class without triggering PyQt6."""
    backends_dir = os.path.join(os.path.dirname(__file__), "workers", "transcription", "backends")
    sys.path.insert(0, backends_dir)

    import types

    for mod_name in [
        "workers",
        "workers.transcription",
        "workers.transcription.backends",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import importlib.util

    base_spec = importlib.util.spec_from_file_location(
        "workers.transcription.backends.base", os.path.join(backends_dir, "base.py")
    )
    base_mod = importlib.util.module_from_spec(base_spec)
    base_mod.__package__ = "workers.transcription.backends"
    sys.modules["workers.transcription.backends.base"] = base_mod
    base_spec.loader.exec_module(base_mod)
    sys.modules["workers.transcription.backends"].TranscriptionBackend = (
        base_mod.TranscriptionBackend
    )

    backend_files = {
        "crispasr": "crispasr_backend.py",
        "crispasr-ffi": "crispasr_ffi_backend.py",
        "whisper.cpp": "whisper_cpp_backend.py",
        "faster-batched": "faster_whisper_backend.py",
        "faster-sequenced": "faster_whisper_backend.py",
        "transformers": "transformers_backend.py",
        "mlx-whisper": "mlx_backend.py",
        "ctranslate2": "ctranslate2_backend.py",
        "whisper-jax": "whisper_jax_backend.py",
        "insanely-fast-whisper": "insanely_fast_backend.py",
        "openai whisper": "openai_whisper_backend.py",
        "voxtral-local": "voxtral_backend.py",
        "voxtral-api": "voxtral_backend.py",
    }
    backend_classes = {
        "crispasr": "CrispasrBackend",
        "crispasr-ffi": "CrispasrFFIBackend",
        "whisper.cpp": "WhisperCppBackend",
        "faster-batched": "FasterWhisperBatchedBackend",
        "faster-sequenced": "FasterWhisperSequencedBackend",
        "transformers": "TransformersBackend",
        "mlx-whisper": "MLXBackend",
        "ctranslate2": "CTranslate2Backend",
        "whisper-jax": "WhisperJaxBackend",
        "insanely-fast-whisper": "InsanelyFastBackend",
        "openai whisper": "OpenAIWhisperBackend",
        "voxtral-local": "VoxtralLocalBackend",
        "voxtral-api": "VoxtralAPIBackend",
    }

    # Handle crispasr-ffi:<subbackend> and crispasr:<subbackend> notation
    lookup_name = name
    if name.startswith("crispasr-ffi:"):
        lookup_name = "crispasr-ffi"
    elif name.startswith("crispasr:"):
        lookup_name = "crispasr"

    if lookup_name not in backend_files:
        raise ValueError(f"Unknown backend: {name}. Available: {', '.join(sorted(backend_files))}")

    fname = backend_files[lookup_name]
    cname = backend_classes[lookup_name]

    spec = importlib.util.spec_from_file_location(
        f"workers.transcription.backends.{fname[:-3]}", os.path.join(backends_dir, fname)
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "workers.transcription.backends"
    spec.loader.exec_module(mod)
    return getattr(mod, cname)


def get_tts_backend_class(name):
    """Import TTS backend class without triggering PyQt6."""
    tts_dir = os.path.join(os.path.dirname(__file__), "workers", "tts", "backends")

    if not os.path.isdir(tts_dir):
        raise ValueError(f"TTS backends directory not found: {tts_dir}")

    import importlib.util
    import types

    for mod_name in [
        "workers.tts",
        "workers.tts.backends",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # Load base
    base_spec = importlib.util.spec_from_file_location(
        "workers.tts.backends.base", os.path.join(tts_dir, "base.py")
    )
    base_mod = importlib.util.module_from_spec(base_spec)
    base_mod.__package__ = "workers.tts.backends"
    sys.modules["workers.tts.backends.base"] = base_mod
    base_spec.loader.exec_module(base_mod)

    tts_backend_files = {
        "crispasr-tts": ("crispasr_tts_backend.py", "CrispasrTTSBackend"),
        "edge-tts": ("edge_tts_backend.py", "EdgeTTSBackend"),
        "piper": ("piper_tts_backend.py", "PiperTTSBackend"),
        "kokoro-onnx": ("kokoro_onnx_tts_backend.py", "KokoroOnnxTTSBackend"),
        "chatterbox": ("chatterbox_tts_backend.py", "ChatterboxTTSBackend"),
        "speecht5": ("speecht5_tts_backend.py", "SpeechT5TTSBackend"),
    }

    # Handle crispasr:<tts-sub> notation
    lookup_name = name
    if name.startswith("crispasr:"):
        lookup_name = "crispasr-tts"

    if lookup_name not in tts_backend_files:
        raise ValueError(
            f"Unknown TTS backend: {name}. Available: {', '.join(sorted(tts_backend_files))}"
        )

    fname, cname = tts_backend_files[lookup_name]
    spec = importlib.util.spec_from_file_location(
        f"workers.tts.backends.{fname[:-3]}", os.path.join(tts_dir, fname)
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "workers.tts.backends"
    spec.loader.exec_module(mod)
    return getattr(mod, cname)


def _read_input_text(args):
    """Get input text from --text or --input-file."""
    if args.text:
        return args.text
    if args.input_file:
        try:
            from utils.text_extraction import extract_text

            return extract_text(args.input_file)
        except ImportError:
            # Fallback: plain text read
            with open(args.input_file, "r", encoding="utf-8") as f:
                return f.read()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Susurrus CLI — transcription, TTS, and translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Mode ---
    parser.add_argument(
        "--mode",
        choices=["transcribe", "tts", "translate", "stream", "server", "align"],
        default="transcribe",
        help="Operation mode (default: transcribe)",
    )

    # --- Common ---
    parser.add_argument("--backend", "-b", default="crispasr", help="Backend (default: crispasr)")
    parser.add_argument("--model", "-m", default=None, help="Model path, HF ID, or 'auto'")
    parser.add_argument("--file", "-f", default=None, help="Audio file to transcribe")
    parser.add_argument("--language", "-l", default=None, help="Language code (e.g. en, de)")
    parser.add_argument("--device", "-d", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--list-backends", action="store_true", help="List all backends and exit")

    # --- Text input (for TTS/translate) ---
    parser.add_argument("--text", default=None, help="Input text (for TTS or translation)")
    parser.add_argument("--input-file", default=None, help="Input text file (TXT/MD/HTML/PDF/EPUB)")

    # --- TTS-specific ---
    tts_group = parser.add_argument_group("TTS Options")
    tts_group.add_argument("--tts-backend", default=None, help="TTS backend override")
    tts_group.add_argument(
        "--tts-output", default="tts_output.wav", help="TTS output file (default: tts_output.wav)"
    )
    tts_group.add_argument("--voice", default=None, help="Voice ID or path to voice file")
    tts_group.add_argument("--ref-text", default=None, help="Reference text for voice cloning")
    tts_group.add_argument(
        "--instruct", default=None, help="Natural-language voice description (qwen3-tts)"
    )
    tts_group.add_argument("--codec-model", default=None, help="Codec/companion GGUF model")
    tts_group.add_argument("--codec-quant", default=None, help="Preferred companion quant")
    tts_group.add_argument("--tts-steps", type=int, default=None, help="TTS diffusion steps")
    tts_group.add_argument("--tts-cfg-scale", type=float, default=None, help="TTS CFG scale")
    tts_group.add_argument(
        "--tts-speed", type=float, default=None, help="TTS speaking-rate multiplier"
    )
    tts_group.add_argument(
        "--tts-trim-silence", action="store_true", help="Trim leading silence from TTS output"
    )
    tts_group.add_argument(
        "--tts-max-input-chars", type=int, default=None, help="Server TTS input length cap"
    )
    tts_group.add_argument("--voice-dir", default=None, help="Directory of named voice profiles")
    tts_group.add_argument("--play", action="store_true", help="Play audio after synthesis")
    tts_group.add_argument(
        "--tts-play", action="store_true", help="Play audio on local speaker (CrispASR native)"
    )
    tts_group.add_argument(
        "--tts-play-device", type=int, default=None, help="Audio device index for local playback"
    )
    tts_group.add_argument("--list-voices", action="store_true", help="List voices for TTS backend")

    # --- EU AI Act / Provenance ---
    prov_group = parser.add_argument_group("Provenance & EU AI Act Compliance")
    prov_group.add_argument(
        "--i-have-rights",
        action="store_true",
        help="Attest voice-cloning consent (REQUIRED for .wav reference cloning)",
    )
    prov_group.add_argument(
        "--no-spoken-disclaimer",
        action="store_true",
        help="Skip audible AI-disclosure prefix (watermark + C2PA still applied)",
    )
    prov_group.add_argument(
        "--watermark-model",
        default=None,
        help="AudioSeal GGUF for neural watermarking (upgrades built-in)",
    )
    prov_group.add_argument(
        "--no-watermark",
        action="store_true",
        help="Disable AI-content watermark — shifts marking responsibility to operator",
    )
    prov_group.add_argument("--no-c2pa", action="store_true", help="Disable C2PA signing")
    prov_group.add_argument(
        "--accept-marking-responsibility",
        action="store_true",
        help="Accept responsibility for AI-content marking when opting out",
    )
    prov_group.add_argument(
        "--accept-license", default=None, help="Accept a restricted model license tag"
    )
    prov_group.add_argument(
        "--detect-watermark",
        default=None,
        help=(
            "Detect an AI watermark in a WAV file and exit. Uses the CrispASR "
            "binary when available and passes through its exit code; "
            "otherwise falls back to the Python detector, which exits 0 if "
            "marked, 1 if not, and 2 if it could not check"
        ),
    )
    prov_group.add_argument("--c2pa-cert", default=None, help="X.509 cert for C2PA signing")
    prov_group.add_argument("--c2pa-key", default=None, help="Private key for C2PA signing")
    prov_group.add_argument(
        "--verify-c2pa",
        default=None,
        help="Verify C2PA credentials in an audio file and exit",
    )
    prov_group.add_argument(
        "--speaker-identity",
        default=None,
        choices=["real_person", "synthetic", "unknown"],
        help=(
            "Whose voice a preset is. 'real_person' makes Susurrus prepend the "
            "audible EU AI Act Art. 50(4) disclosure to stock-voice output too, "
            "not only to cloned voices. Overrides the shipped classification"
        ),
    )
    prov_group.add_argument(
        "--about-ai",
        action="store_true",
        help=(
            "Print the AI-literacy notice (EU AI Act Art. 4) and exit: what "
            "this system is, its intended purpose, its failure modes, and what "
            "it is not validated for"
        ),
    )

    # --- Translation-specific ---
    tr_group = parser.add_argument_group("Translation Options")
    tr_group.add_argument("--source-lang", default=None, help="Source language code")
    tr_group.add_argument("--target-lang", default=None, help="Target language code")
    tr_group.add_argument(
        "--translate-max-tokens", type=int, default=None, help="Max output tokens"
    )

    # --- CrispASR pass-through ---
    ca_group = parser.add_argument_group("CrispASR Options")
    ca_group.add_argument("--crispasr-backend", default=None, help="Force CrispASR sub-backend")
    ca_group.add_argument("--diagnostics", action="store_true", help="Run CrispASR diagnostics")
    ca_group.add_argument("--vad", action="store_true", help="Enable VAD")
    ca_group.add_argument("--split-on-punct", action="store_true", help="Split at punctuation")
    ca_group.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    ca_group.add_argument("--best-of", type=int, default=None, help="Best-of-N candidates")
    ca_group.add_argument("--beam-size", type=int, default=None, help="Beam search width")
    ca_group.add_argument("--seed", type=int, default=None, help="RNG seed")
    ca_group.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens")
    ca_group.add_argument("--frequency-penalty", type=float, default=None, help="Frequency penalty")
    ca_group.add_argument("--prompt", default=None, help="Initial prompt")
    ca_group.add_argument(
        "--carry-initial-prompt", action="store_true", help="Always prepend initial prompt"
    )
    ca_group.add_argument("--auto-download", action="store_true", help="Auto-download model")
    ca_group.add_argument(
        "--prefix-text", default=None, help="LLM initial prompt (granite keyword biasing)"
    )
    ca_group.add_argument("--context", default=None, help="Context text for supported backends")
    ca_group.add_argument("--translate", action="store_true", help="Translate to English (whisper)")
    ca_group.add_argument("--flash-attn", action="store_true", help="Enable flash attention")
    ca_group.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    ca_group.add_argument("--gpu-backend", default=None, help="GPU backend (cuda/vulkan/metal)")
    ca_group.add_argument(
        "--n-gpu-layers", type=int, default=None, help="GPU layer offload count (-1 = all)"
    )
    ca_group.add_argument(
        "--no-kv-offload", action="store_true", help="Keep KV cache on CPU, weights on GPU"
    )

    # --- CrispASR VAD options ---
    vad_group = parser.add_argument_group("CrispASR VAD Options")
    vad_group.add_argument("--vad-model", default=None, help="VAD model (firered/silero/path)")
    vad_group.add_argument("--vad-threshold", type=float, default=None, help="VAD threshold 0-1")
    vad_group.add_argument(
        "--vad-min-speech-ms", type=int, default=None, help="Min speech duration (ms)"
    )
    vad_group.add_argument(
        "--vad-min-silence-ms", type=int, default=None, help="Min silence duration (ms)"
    )
    vad_group.add_argument(
        "--vad-max-speech-s", type=float, default=None, help="Max speech duration (s)"
    )
    vad_group.add_argument("--vad-pad-ms", type=int, default=None, help="Speech pad (ms)")
    vad_group.add_argument("--vad-export", default=None, help="Write VAD/chunk boundaries to JSON")
    vad_group.add_argument("--vad-import", default=None, help="Read VAD/chunk boundaries from JSON")
    vad_group.add_argument(
        "--vad-import-strict", action="store_true", help="Require imported VAD metadata to match"
    )
    vad_group.add_argument("--vad-export-raw", default=None, help="Write raw VAD speech segments")

    # --- CrispASR diarization ---
    dia_group = parser.add_argument_group("CrispASR Diarization Options")
    dia_group.add_argument("--diarize", action="store_true", help="Enable diarization")
    dia_group.add_argument(
        "--diarize-method",
        default=None,
        help="Method: energy/xcorr/vad-turns/pyannote/sherpa/ecapa",
    )
    dia_group.add_argument("--diarize-embedder", default=None, help="Speaker embedder model")
    dia_group.add_argument(
        "--diarize-cluster-threshold", type=float, default=None, help="Cluster merge threshold"
    )
    dia_group.add_argument(
        "--diarize-max-speakers", type=int, default=None, help="Max speaker count"
    )
    dia_group.add_argument(
        "--diarize-speakers", action="store_true", help="Enable diarization with auto embedder"
    )
    dia_group.add_argument(
        "--speaker-db-consent",
        action="store_true",
        help=(
            "Attest a lawful basis for storing voice biometrics linked to "
            "named people (GDPR Art. 9 special-category data). Required for "
            "--speaker-db and --enroll-speaker"
        ),
    )

    # --- CrispASR LID ---
    lid_group = parser.add_argument_group("CrispASR Language ID Options")
    lid_group.add_argument(
        "--detect-language", action="store_true", help="Detect language and exit"
    )
    lid_group.add_argument(
        "--lid-backend", default=None, help="LID method: whisper/silero/firered/ecapa"
    )
    lid_group.add_argument("--lid-model", default=None, help="Custom LID model")

    # --- CrispASR alignment ---
    align_group = parser.add_argument_group("CrispASR Alignment Options")
    align_group.add_argument("--aligner-model", default=None, help="CTC aligner GGUF")
    align_group.add_argument("--force-aligner", action="store_true", help="Force CTC alignment")
    align_group.add_argument(
        "--text-file", default=None, help="Text/SRT file for --align-only mode"
    )
    align_group.add_argument("--align-output", default=None, help="Alignment output path")
    align_group.add_argument(
        "--align-format",
        default=None,
        choices=["srt", "json", "plain"],
        help="Alignment output format",
    )

    # --- CrispASR punctuation ---
    punc_group = parser.add_argument_group("CrispASR Punctuation Options")
    punc_group.add_argument("--punc-model", default=None, help="Punctuation restoration model")

    # --- CrispASR speaker ---
    spk_group = parser.add_argument_group("CrispASR Speaker Options")
    spk_group.add_argument("--speaker-db", default=None, help="Speaker profile database path")
    spk_group.add_argument(
        "--expect-speakers", default=None, help="Comma-separated enrolled speakers"
    )
    spk_group.add_argument("--enroll-speaker", default=None, help="Enroll speaker name")
    spk_group.add_argument("--speaker-threshold", type=float, default=None, help="Match threshold")
    spk_group.add_argument("--titanet-model", default=None, help="Speaker embedding model")
    spk_group.add_argument(
        "--audit-log",
        action="store_true",
        help=(
            "Print the EU AI Act Art. 12 biometric audit log and verify its "
            "hash chain, then exit (exit 1 if the chain is broken)"
        ),
    )

    # --- CrispASR audio analysis ---
    analysis_group = parser.add_argument_group("CrispASR Audio Analysis Options")
    analysis_group.add_argument("--separate", action="store_true", help="Run source separation")
    analysis_group.add_argument("--stems", default=None, help="Comma-separated stems to write")
    analysis_group.add_argument(
        "--sep-output-dir", default=None, help="Source separation output dir"
    )
    analysis_group.add_argument("--pitch", action="store_true", help="Run pitch tracking")
    analysis_group.add_argument("--pitch-format", default=None, choices=["text", "json"])
    analysis_group.add_argument("--pitch-hop-ms", type=float, default=None)
    analysis_group.add_argument("--piano", action="store_true", help="Run piano transcription")
    analysis_group.add_argument("--piano-format", default=None, choices=["text", "json"])
    analysis_group.add_argument("--chords", action="store_true", help="Run chord recognition")
    analysis_group.add_argument("--chords-format", default=None, choices=["text", "json"])
    analysis_group.add_argument("--tab", action="store_true", help="Run guitar tablature")
    analysis_group.add_argument("--tab-format", default=None, choices=["text", "json"])
    analysis_group.add_argument("--beats", action="store_true", help="Run beat tracking")
    analysis_group.add_argument("--beats-format", default=None, choices=["text", "json"])

    # --- CrispASR grammar ---
    gram_group = parser.add_argument_group("CrispASR Grammar Options")
    gram_group.add_argument("--grammar", default=None, help="GBNF grammar for constrained decoding")
    gram_group.add_argument("--grammar-rule", default=None, help="Top-level grammar rule")
    gram_group.add_argument("--grammar-penalty", type=float, default=None, help="Grammar penalty")

    # --- CrispASR output ---
    out_group = parser.add_argument_group("CrispASR Output Format Options")
    out_group.add_argument("--output-srt", action="store_true", help="Output SRT subtitles")
    out_group.add_argument("--output-vtt", action="store_true", help="Output WebVTT subtitles")
    out_group.add_argument("--output-json", action="store_true", help="Output JSON")
    out_group.add_argument("--output-json-full", action="store_true", help="Output full JSON")
    out_group.add_argument("--output-csv", action="store_true", help="Output CSV")
    out_group.add_argument("--output-lrc", action="store_true", help="Output LRC lyrics")
    out_group.add_argument("--output-file", default=None, help="Output file base path")

    # --- CrispASR streaming ---
    stream_group = parser.add_argument_group("CrispASR Streaming Options")
    stream_group.add_argument("--mic", action="store_true", help="Capture from microphone")
    stream_group.add_argument("--live", action="store_true", help="Continuous live transcription")
    stream_group.add_argument("--stream-step", type=int, default=None, help="Chunk size (ms)")
    stream_group.add_argument("--stream-length", type=int, default=None, help="Context window (ms)")
    stream_group.add_argument("--stream-json", action="store_true", help="JSON-Lines output")

    # --- CrispASR server ---
    srv_group = parser.add_argument_group("CrispASR Server Options")
    srv_group.add_argument("--host", default=None, help="Server bind address")
    srv_group.add_argument("--port", type=int, default=None, help="Server port")
    srv_group.add_argument("--api-keys", default=None, help="Comma-separated API keys")
    srv_group.add_argument(
        "--wyoming-port", type=int, default=None, help="Wyoming protocol TCP port (Home Assistant)"
    )

    # --- Misc ---
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--output-format",
        choices=["txt", "srt", "vtt", "json", "csv"],
        default=None,
        help="Output format for transcription (default: timestamped text to stdout)",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if args.list_backends:
        _list_backends()
        return

    if getattr(args, "diagnostics", False):
        from utils.crispasr_utils import find_crispasr

        exe = find_crispasr()
        if not exe:
            print("crispasr binary not found", file=sys.stderr)
            sys.exit(1)
        import subprocess

        proc = subprocess.run([exe, "--diagnostics"], text=True)
        sys.exit(proc.returncode)

    # EU AI Act compliance warnings
    if getattr(args, "no_watermark", False):
        logging.warning(
            "Watermarking disabled (--no-watermark). AI-content marking "
            "responsibility rests with the operator per EU AI Act Art. 50."
        )

    # The warning is advisory and belongs up front, before any work starts.
    # The Art. 12 *record* is written after the run instead — see
    # _audit_speaker_biometrics — because a log entry made here would assert
    # that people were enrolled or identified even when the run then failed,
    # or never processed audio at all.
    _warn_speaker_biometrics(args)

    # --about-ai is a standalone verb: the Art. 4 AI-literacy notice. The GUI
    # carries this under Help > About AI in Susurrus, and CLI-only deployments
    # had no equivalent — so the one obligation that is about people rather
    # than files was reachable only by the users who never opened the GUI.
    # Rendered from the same localized source the dialog uses, so the two
    # cannot drift apart.
    if getattr(args, "about_ai", False):
        from utils.i18n import t

        print(_html_to_text(t("msg.ai_notice.body")))
        sys.exit(0)

    # --audit-log is a standalone verb: read and verify the Art. 12 record
    if getattr(args, "audit_log", False):
        import json

        from utils.audit_log import audit_log_path, read_events, verify_chain

        events = read_events()
        chain = verify_chain()
        print(json.dumps({"path": audit_log_path(), "chain": chain, "events": events}, indent=2))
        sys.exit(0 if chain["valid"] else 1)

    # --verify-c2pa is a standalone verb (Python-side, no binary needed).
    # It reports *both* marking layers: a file carrying only the declarative
    # marker is still marked as AI-generated, and answering "is this marked?"
    # with exit 1 just because c2pa-audio is missing would be misleading.
    #
    # Dispatch on the container, as the marking side does. The WAV-only
    # readers cannot see an ID3 marker, so verifying a correctly marked MP3
    # reported "not AI-generated" and exited 1 — Susurrus contradicting its
    # own output. edge-tts synthesizes MP3 natively and the GUI save dialog
    # offers .mp3, so this was the common case, not an edge one.
    if getattr(args, "verify_c2pa", None):
        try:
            import json

            from utils.ai_marking import read_ai_marker
            from utils.c2pa_signing import verify_audio_file

            c2pa_result = verify_audio_file(args.verify_c2pa)
            marker = read_ai_marker(args.verify_c2pa)

            report = {
                "c2pa": c2pa_result if c2pa_result else {"available": c2pa_result is not None},
                "ai_marker": marker or None,
                "marked_as_ai_generated": bool(
                    (c2pa_result and c2pa_result.get("valid")) or marker
                ),
            }
            print(json.dumps(report, indent=2))
            sys.exit(0 if report["marked_as_ai_generated"] else 1)
        except Exception as e:
            print(f"Provenance verification error: {e}", file=sys.stderr)
            sys.exit(1)

    # --detect-watermark is a standalone verb. Prefer the CrispASR binary when
    # available; fall back to the Python AudioSeal detector so the verb works
    # on installs that have no binary.
    if getattr(args, "detect_watermark", None):
        sys.exit(_run_detect_watermark(args))

    if args.mode == "transcribe":
        _run_transcribe(args)
    elif args.mode == "tts":
        _run_tts(args)
    elif args.mode == "translate":
        _run_translate(args)
    elif args.mode == "stream":
        _run_stream(args)
    elif args.mode == "server":
        _run_server(args)
    elif args.mode == "align":
        _run_align(args)


def _list_backends():
    """List all available backends."""
    print("Transcription backends:")
    transcription = [
        "crispasr",
        "crispasr-ffi",
        "faster-batched",
        "faster-sequenced",
        "transformers",
        "whisper.cpp",
        "ctranslate2",
        "whisper-jax",
        "insanely-fast-whisper",
        "openai whisper",
        "voxtral-local",
        "voxtral-api",
        "mlx-whisper",
    ]
    for b in transcription:
        print(f"  {b}")

    print("\nCrispASR ASR sub-backends (use as crispasr:<name>):")
    from config import CRISPASR_SUB_BACKENDS

    for b in CRISPASR_SUB_BACKENDS:
        print(f"  crispasr:{b}")

    print("\nTTS backends:")
    tts = [
        "edge-tts",
        "piper",
        "kokoro-onnx",
        "chatterbox",
        "speecht5",
    ]
    for b in tts:
        print(f"  {b}")

    print("\nCrispASR TTS backends (use as crispasr:<name>):")
    from config import CRISPASR_TTS_BACKENDS

    for b in CRISPASR_TTS_BACKENDS:
        print(f"  crispasr:{b}")

    print("\nTranslation backends:")
    from config import CRISPASR_TRANSLATION_BACKENDS

    for b in CRISPASR_TRANSLATION_BACKENDS:
        print(f"  crispasr:{b}")


def _build_crispasr_kwargs(args):
    """Build kwargs dict for CrispASR backend from parsed CLI args."""
    kwargs = {}

    # Map CLI arg names to CrispASR kwarg names
    mappings = {
        "crispasr_backend": "crispasr_backend",
        "diagnostics": "diagnostics",
        "vad": "vad",
        "split_on_punct": "split_on_punct",
        "temperature": "temperature",
        "best_of": "best_of",
        "beam_size": "beam_size",
        "seed": "seed",
        "max_new_tokens": "max_new_tokens",
        "frequency_penalty": "frequency_penalty",
        "prompt": "prompt",
        "carry_initial_prompt": "carry_initial_prompt",
        "auto_download": "auto_download",
        "prefix_text": "prefix_text",
        "context": "context",
        "translate": "translate",
        "flash_attn": "flash_attn",
        "no_gpu": "no_gpu",
        "gpu_backend": "gpu_backend",
        "n_gpu_layers": "n_gpu_layers",
        "no_kv_offload": "no_kv_offload",
        # VAD
        "vad_model": "vad_model",
        "vad_threshold": "vad_threshold",
        "vad_min_speech_ms": "vad_min_speech_duration_ms",
        "vad_min_silence_ms": "vad_min_silence_duration_ms",
        "vad_max_speech_s": "vad_max_speech_duration_s",
        "vad_pad_ms": "vad_speech_pad_ms",
        "vad_export": "vad_export",
        "vad_import": "vad_import",
        "vad_import_strict": "vad_import_strict",
        "vad_export_raw": "vad_export_raw",
        # Diarization
        "diarize": "diarize",
        "diarize_method": "diarize_method",
        "diarize_embedder": "diarize_embedder",
        "diarize_cluster_threshold": "diarize_cluster_threshold",
        "diarize_max_speakers": "diarize_max_speakers",
        "diarize_speakers": "diarize_speakers",
        "speaker_db_consent": "speaker_db_consent",
        # LID
        "detect_language": "detect_language",
        "lid_backend": "lid_backend",
        "lid_model": "lid_model",
        # Alignment
        "aligner_model": "aligner_model",
        "force_aligner": "force_aligner",
        "text_file": "text_file",
        "align_output": "align_output",
        "align_format": "align_format",
        # Punctuation
        "punc_model": "punc_model",
        # Speaker
        "speaker_db": "speaker_db",
        "expect_speakers": "expect_speakers",
        "enroll_speaker": "enroll_speaker",
        "speaker_threshold": "speaker_threshold",
        "titanet_model": "titanet_model",
        # Grammar
        "grammar": "grammar",
        "grammar_rule": "grammar_rule",
        "grammar_penalty": "grammar_penalty",
        # Output
        "output_srt": "output_srt",
        "output_vtt": "output_vtt",
        "output_json": "output_json",
        "output_json_full": "output_json_full",
        "output_csv": "output_csv",
        "output_lrc": "output_lrc",
        "output_file": "output_file",
        # Streaming
        "mic": "mic",
        "live": "live",
        "stream_step": "stream_step",
        "stream_length": "stream_length",
        "stream_json": "stream_json",
        # Server
        "host": "host",
        "port": "port",
        "api_keys": "api_keys",
        "wyoming_port": "wyoming_port",
        # TTS
        "voice": "tts_voice",
        "ref_text": "tts_ref_text",
        "instruct": "tts_instruct",
        "codec_model": "tts_codec_model",
        "codec_quant": "tts_codec_quant",
        "tts_steps": "tts_steps",
        "tts_cfg_scale": "tts_cfg_scale",
        "tts_speed": "tts_speed",
        "tts_trim_silence": "tts_trim_silence",
        "tts_max_input_chars": "tts_max_input_chars",
        "voice_dir": "voice_dir",
        "tts_play": "tts_play",
        "tts_play_device": "tts_play_device",
        # Provenance / EU AI Act
        "i_have_rights": "i_have_rights",
        "accept_license": "accept_license",
        "no_spoken_disclaimer": "no_spoken_disclaimer",
        "watermark_model": "watermark_model",
        "no_watermark": "no_watermark",
        "no_c2pa": "no_c2pa",
        "accept_marking_responsibility": "accept_marking_responsibility",
        "detect_watermark": "detect_watermark",
        "c2pa_cert": "c2pa_cert",
        "c2pa_key": "c2pa_key",
        # Audio analysis / source separation
        "separate": "separate",
        "stems": "stems",
        "sep_output_dir": "sep_output_dir",
        "pitch": "pitch",
        "pitch_format": "pitch_format",
        "pitch_hop_ms": "pitch_hop_ms",
        "piano": "piano",
        "piano_format": "piano_format",
        "chords": "chords",
        "chords_format": "chords_format",
        "tab": "tab",
        "tab_format": "tab_format",
        "beats": "beats",
        "beats_format": "beats_format",
    }

    # Handle crispasr:<sub> notation
    backend = args.backend
    if backend.startswith("crispasr:"):
        sub = backend.split(":", 1)[1]
        kwargs["crispasr_backend"] = sub

    for arg_name, kwarg_name in mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None and value is not False:
            kwargs[kwarg_name] = value

    return kwargs


def _run_transcribe(args):
    """Run transcription mode."""
    if not args.file:
        print("Error: --file is required for transcription", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    backend_name = args.backend
    model = args.model

    # Default model for CrispASR backends
    if not model and backend_name.startswith("crispasr"):
        model = "auto"

    if not model:
        print("Error: --model is required for transcription", file=sys.stderr)
        sys.exit(1)

    kwargs = _build_crispasr_kwargs(args)

    BackendClass = get_backend_class(backend_name)
    backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)

    try:
        audio_path = backend.preprocess_audio(args.file)
        segments = list(backend.transcribe(audio_path))

        # Art. 12: record the biometric event now that it has actually
        # happened, rather than when the flags were parsed.
        _audit_speaker_biometrics(args)

        output_format = getattr(args, "output_format", None)
        if output_format:
            from utils.export_formats import EXPORT_FORMATS

            fmt_name = output_format.upper()
            if fmt_name in EXPORT_FORMATS:
                _, export_fn = EXPORT_FORMATS[fmt_name]
                print(export_fn(segments))
            else:
                print(f"Unknown format: {output_format}", file=sys.stderr)
        else:
            for start, end, text in segments:
                if start > 0 or end > 0:
                    print(f"[{start:.2f} --> {end:.2f}]  {text}")
                else:
                    print(text)
    finally:
        backend.cleanup()


def _html_to_text(html):
    """Render the localized AI-literacy notice as plain text for a terminal.

    The notice is authored once, as HTML, for the GUI dialog. Keeping a second
    plain-text copy in the translation files would mean two texts to keep in
    step across every language — and the one that drifts is always the one
    nobody is looking at.
    """
    import re

    text = re.sub(r"</(h[1-6]|p|ul)>", "\n\n", html)
    text = re.sub(r"<li>", "  - ", text)
    text = re.sub(r"</li>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # The source is trusted (our own translation files), but it is still HTML:
    # decode entities so "&amp;" does not reach the terminal as written.
    import html as html_module

    return html_module.unescape(text).strip()


def _warn_speaker_biometrics(args):
    """Warn when the speaker database is used without a consent attestation.

    Enrolling a named speaker stores a voice embedding — biometric data under
    GDPR Art. 9, and potentially an Annex III(1)(a) high-risk use under the
    EU AI Act depending on how the deployment identifies people.

    Returns True if a warning was emitted.
    """
    uses_biometrics = bool(
        getattr(args, "speaker_db", None)
        or getattr(args, "enroll_speaker", None)
        or getattr(args, "expect_speakers", None)
    )
    if not uses_biometrics or getattr(args, "speaker_db_consent", False):
        return False

    logging.warning(
        "Speaker database in use without --speaker-db-consent. Enrolling a "
        "named speaker stores voice biometrics (GDPR Art. 9 special-category "
        "data). Confirm you have a lawful basis and the speaker's consent. "
        "Identifying people this way may also be a high-risk use under EU AI "
        "Act Annex III(1)(a) — see COMPLIANCE.md."
    )
    return True


def _audit_speaker_biometrics(args):
    """Record biometric events to the Art. 12 audit log.

    Enrollment and identification are logged separately: Art. 12 covers use of
    the system, not only its setup, so a deployer must be able to show when
    people were *matched* against the database as well as added to it.

    Call this **after** the run, not while parsing arguments. A record written
    from the flags alone documents an intention rather than an event: it
    claims an identification happened even when the backend failed to start,
    the audio was unreadable, or the user aborted. An audit trail that
    overstates what occurred is worse than a sparse one.

    Returns the list of entries written.
    """
    enroll = getattr(args, "enroll_speaker", None)
    database = getattr(args, "speaker_db", None)
    identifies = bool(getattr(args, "expect_speakers", None) or (database and not enroll))
    if not enroll and not identifies:
        return []

    from utils.audit_log import EVENT_ENROLL, EVENT_IDENTIFY, record_event

    common = {
        "database": database,
        "consent": getattr(args, "speaker_db_consent", False),
        "model": getattr(args, "titanet_model", None),
    }

    written = []
    if enroll:
        written.append(record_event(EVENT_ENROLL, speaker=enroll, **common))
    if identifies:
        written.append(record_event(EVENT_IDENTIFY, speaker=None, **common))
    return [e for e in written if e]


def _detect_watermark_via_binary(args, target):
    """Run the CrispASR detector. Returns its exit code, or None if unusable.

    None means "fall back to the Python detector": either no binary is
    installed, or the one that is installed failed to run at all (a stale
    build, a missing dylib). A broken binary should not turn "is this AI
    audio?" into an unanswered question when a Python detector is available.
    """
    from utils.crispasr_utils import find_crispasr

    if not find_crispasr():
        return None

    kwargs = _build_crispasr_kwargs(args)
    from workers.transcription.backends.crispasr_backend import CrispasrBackend

    backend = CrispasrBackend(model_id="auto", device="cpu", **kwargs)
    try:
        cmd, _ = backend._build_base_cmd()
        cmd.extend(["--detect-watermark", target])
        backend._append_params(cmd)
        import subprocess

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            logging.warning("CrispASR watermark detector could not run: %s", e)
            return None

        if proc.returncode != 0 and not proc.stdout.strip():
            # Crashed before producing a verdict — not a "no watermark" answer.
            logging.warning(
                "CrispASR watermark detector failed (exit %d); falling back to "
                "the Python detector. %s",
                proc.returncode,
                proc.stderr.strip().splitlines()[0] if proc.stderr.strip() else "",
            )
            return None

        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode
    finally:
        backend.cleanup()


def _run_detect_watermark(args):
    """Detect an AI watermark in a file. Returns the process exit code.

    Tries the CrispASR binary first (it also knows about its own watermark
    variants), then the Python AudioSeal detector. Reports which detector ran
    so an inconclusive answer is not mistaken for a negative one.
    """
    target = args.detect_watermark

    binary_result = _detect_watermark_via_binary(args, target)
    if binary_result is not None:
        return binary_result

    import json

    # read_ai_marker dispatches on the container; the WAV-only reader silently
    # missed the ID3 marker that MP3 output actually carries.
    from utils.ai_marking import read_ai_marker
    from utils.audio_watermark import detect_watermark

    neural = detect_watermark(target)
    marker = read_ai_marker(target)

    # Two independent checks. Name them separately: the declarative marker is
    # always readable, the in-sample detector may be unavailable, and a report
    # that collapses them makes "could not check" look like "clean".
    #
    # Name the detector that actually ran, too. detect_watermark() falls back
    # to the spread-spectrum tier when AudioSeal is absent, so labelling every
    # result "audioseal" misreported which scheme produced the verdict — and
    # the two differ in what they are robust against.
    report = {
        "neural_detector": (neural.get("backend", "unknown") if neural else "unavailable"),
        "neural_watermark": neural,
        "ai_marker": marker or None,
        "detected_as_ai_generated": bool((neural and neural["watermarked"]) or marker),
    }
    print(json.dumps(report, indent=2))

    if neural is None and marker is None:
        print(
            "Inconclusive: no in-sample detector available (install the "
            "crispasr binary, or 'pip install audioseal', or the numpy + "
            "soundfile stack for the spread-spectrum detector) and no "
            "declarative marker found. This is 'could not check', not "
            "'not AI-generated'.",
            file=sys.stderr,
        )
        return 2

    return 0 if report["detected_as_ai_generated"] else 1


#: Provenance opt-outs that require an explicit attestation to take effect.
_MARKING_OPT_OUTS = (
    ("no_watermark", "--no-watermark"),
    ("no_c2pa", "--no-c2pa"),
    ("no_spoken_disclaimer", "--no-spoken-disclaimer"),
)


def _require_marking_attestation(args):
    """Refuse a provenance opt-out that is not backed by an attestation.

    The CrispASR binary requires ``--accept-marking-responsibility`` before it
    honours any of these, and force-keeps the watermark on the CLI otherwise.
    Susurrus used to honour them unconditionally on the Python-native path, so
    the same flag meant different things depending on the backend. Refusing
    early makes one rule, and makes the operator state that they are taking
    the Art. 50 obligation on.
    """
    if getattr(args, "accept_marking_responsibility", False):
        return

    used = [flag for attr, flag in _MARKING_OPT_OUTS if getattr(args, attr, False)]
    if not used:
        return

    print(
        f"Refused: {', '.join(used)} reduces EU AI Act Art. 50 provenance and "
        "requires --accept-marking-responsibility, which attests that "
        "responsibility for marking and disclosing this output rests with you "
        "as the operator.",
        file=sys.stderr,
    )
    sys.exit(2)


def _report_disclosure_shortfall(marking):
    """Warn when a cloning run owed an audible disclosure and did not give one.

    Art. 50(4) is a duty to disclose to whoever hears the audio. Machine-
    readable marking does not reach a listener, so it cannot stand in for it.
    """
    from utils.provenance import disclosure_missing

    if not disclosure_missing(marking):
        return

    print(
        "WARNING: this audio clones a voice but carries no audible "
        "disclosure. EU AI Act Art. 50(4) requires disclosure that deepfake "
        "content is artificially generated, and the machine-readable marking "
        "below does not discharge it for a listener.",
        file=sys.stderr,
    )


def _report_marking(marking):
    """Print the EU AI Act Art. 50 provenance status for synthesized audio.

    Reports the two obligations separately. Art. 50(2) marking and the
    Art. 50(4) audible disclosure can succeed and fail independently, and
    folding them into one verdict meant a cloned voice that announced nothing
    to a listener still printed a confident "Marked as AI-generated".
    """
    _report_disclosure_shortfall(marking)

    if marking.get("opted_out"):
        layers = [
            label
            for key, label in (
                ("watermark", "watermark"),
                ("marker", "AI marker"),
                ("c2pa", "C2PA"),
            )
            if marking.get(key)
        ]
        detail = f" The backend still applied: {' + '.join(layers)}." if layers else ""
        print(
            "AI-content marking skipped. Responsibility for marking this "
            f"output rests with the operator per EU AI Act Art. 50.{detail}",
            file=sys.stderr,
        )
        return
    layers = [
        label
        for key, label in (
            ("spoken", "spoken disclosure"),
            ("watermark", "watermark"),
            ("marker", "AI marker"),
            ("c2pa", "C2PA"),
        )
        if marking.get(key)
    ]
    if layers:
        print(f"Marked as AI-generated ({' + '.join(layers)}).")
    else:
        # Reached only when marking is skipped rather than failed: a genuine
        # failure raises ProvenanceError and deletes the file before getting
        # here. Kept because ``opted_out`` returns above and a caller that
        # constructs a bare result should still see something honest.
        print(
            "WARNING: this audio carries no AI-generation marking. EU AI Act "
            "Art. 50(2) requires machine-readable marking of synthetic audio.",
            file=sys.stderr,
        )


def _preflight_marking(args, output_path, is_cloning):
    """Refuse before synthesis when this install cannot mark the output.

    ``enforce_marking`` is the check that counts, but it can only run after
    the audio exists — by which point the user has waited through a model load
    and a synthesis for output that is about to be deleted. This asks the
    knowable part up front, in milliseconds, so a refusal costs nothing.
    """
    if getattr(args, "accept_marking_responsibility", False):
        return

    from utils.provenance import marking_available

    ok, reason = marking_available(output_path, is_cloning=is_cloning)
    if ok:
        return

    print(f"Refused: {reason}", file=sys.stderr)
    print(
        "Susurrus does not emit unmarked synthetic audio. Pass "
        "--accept-marking-responsibility to take the EU AI Act Art. 50 "
        "obligation on yourself, or use a .wav / .mp3 output path.",
        file=sys.stderr,
    )
    sys.exit(2)


def _run_tts(args):
    """Run TTS mode."""
    text = _read_input_text(args)
    if not text:
        print("Error: --text or --input-file is required for TTS", file=sys.stderr)
        sys.exit(1)

    tts_backend = args.tts_backend or args.backend
    output_path = args.tts_output

    _require_marking_attestation(args)

    # A path-like --voice is reference audio, which engages Art. 50(4) and so
    # needs the disclosure path to be available too.
    _preflight_marking(
        args, output_path, is_cloning=bool(args.voice and os.path.isfile(args.voice))
    )

    # Route to CrispASR TTS or Python TTS backend
    if tts_backend.startswith("crispasr"):
        model = args.model or "auto"
        kwargs = _build_crispasr_kwargs(args)
        # The backend has to know its own name to answer two questions: whose
        # voice a preset is, and whether --voice selects from a baked voice
        # bank (which is a clone that never touches the filesystem). Without
        # this the CrispASR TTS route could answer neither.
        kwargs["tts_backend_name"] = tts_backend
        kwargs["speaker_identity"] = getattr(args, "speaker_identity", None)
        BackendClass = get_backend_class(tts_backend)
        backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)
        try:
            result = backend.synthesize(text, output_path, voice=args.voice)
            _report_marking(backend.apply_provenance(result, model=model, voice=args.voice))
            print(f"Audio saved to: {result}")
        except PermissionError as e:
            # Voice-cloning consent gate — a refusal, not a crash.
            print(f"Refused: {e}", file=sys.stderr)
            sys.exit(2)
        except ProvenanceError as e:
            # Art. 50 gate — the unmarked output has already been deleted.
            print(f"Refused: {e}", file=sys.stderr)
            sys.exit(2)
        finally:
            backend.cleanup()
    else:
        TTSClass = get_tts_backend_class(tts_backend)
        # Provenance kwargs must reach the Python-native backends too — they
        # gate cloning and mark output via TTSBackend, not via binary flags.
        # no_watermark and no_spoken_disclaimer were missing here, so both
        # documented flags silently did nothing on every Python backend.
        tts_kwargs = {
            "i_have_rights": getattr(args, "i_have_rights", False),
            "no_watermark": getattr(args, "no_watermark", False),
            "no_spoken_disclaimer": getattr(args, "no_spoken_disclaimer", False),
            "no_c2pa": getattr(args, "no_c2pa", False),
            "accept_marking_responsibility": getattr(args, "accept_marking_responsibility", False),
            "c2pa_cert": getattr(args, "c2pa_cert", None),
            "c2pa_key": getattr(args, "c2pa_key", None),
            # Decides whether a *preset* voice owes the Art. 50(4) disclosure.
            "speaker_identity": getattr(args, "speaker_identity", None),
            "tts_backend_name": tts_backend,
        }
        if args.voice:
            tts_kwargs["voice"] = args.voice
        backend = TTSClass(
            model_id=args.model, device=args.device, language=args.language, **tts_kwargs
        )
        try:
            if args.list_voices:
                voices = backend.list_voices()
                for v in voices:
                    print(f"  {v}")
                return
            result = backend.synthesize(text, output_path, voice=args.voice)
            _report_marking(backend.apply_provenance(result, model=args.model, voice=args.voice))
            print(f"Audio saved to: {result}")
        except PermissionError as e:
            # Voice-cloning consent gate — a refusal, not a crash.
            print(f"Refused: {e}", file=sys.stderr)
            sys.exit(2)
        except ProvenanceError as e:
            # Art. 50 gate — the unmarked output has already been deleted.
            print(f"Refused: {e}", file=sys.stderr)
            sys.exit(2)
        finally:
            backend.cleanup()


def _run_translate(args):
    """Run translation mode."""
    text = _read_input_text(args)
    if not text:
        print("Error: --text or --input-file is required for translation", file=sys.stderr)
        sys.exit(1)

    backend_name = args.backend
    model = args.model or "auto"
    kwargs = _build_crispasr_kwargs(args)

    BackendClass = get_backend_class(backend_name)
    backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)

    try:
        result = backend.translate_text(
            text,
            source_lang=args.source_lang or "en",
            target_lang=args.target_lang or "de",
        )
        print(result)
        _disclose_synthetic_text()
    finally:
        backend.cleanup()


def _disclose_synthetic_text():
    """Tell the operator the translation is machine-generated, on stderr.

    Art. 50(2) names synthetic *text* alongside audio, and Susurrus does not
    mark translation output — the "assistive function for standard editing"
    exemption is the better reading for transforming text a user supplied, and
    COMPLIANCE.md argues it. But that reading covers *marking*; it does not make
    the output any less machine-generated, and the Art. 50(4) duty on text
    published to inform the public on a matter of public interest still lands
    on whoever publishes it.

    So: say so, and say it on stderr. Putting the notice on stdout would
    corrupt the payload for the pipelines and redirects this mode exists to
    feed — a disclosure that makes the output unusable gets suppressed, and a
    suppressed disclosure discloses nothing.
    """
    print(
        "NOTE: this text was produced by a machine-translation model. Machine "
        "translation loses nuance and can invert meaning, especially around "
        "negation, idiom and ambiguous pronouns — have a person review it "
        "before relying on it. If you publish it to inform the public on a "
        "matter of public interest, the EU AI Act Art. 50(4) duty to disclose "
        "that the text is artificially generated is yours. See COMPLIANCE.md.",
        file=sys.stderr,
    )


def _run_stream(args):
    """Run streaming mode."""
    backend_name = args.backend
    model = args.model or "auto"
    kwargs = _build_crispasr_kwargs(args)
    kwargs["stream"] = True

    if not backend_name.startswith("crispasr"):
        print("Error: streaming is only supported with crispasr backends", file=sys.stderr)
        sys.exit(1)

    BackendClass = get_backend_class(backend_name)
    backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)

    try:
        # Streaming uses the binary directly — output comes from stdout
        from utils.crispasr_utils import find_crispasr

        exe = find_crispasr()
        if not exe:
            print("Error: crispasr binary not found", file=sys.stderr)
            sys.exit(1)

        cmd = [exe, "-m", model]
        if args.language:
            cmd.extend(["-l", args.language])
        if args.mic:
            cmd.append("--mic")
        if args.live:
            cmd.append("--live")
        else:
            cmd.append("--stream")
        if args.stream_step:
            cmd.extend(["--stream-step", str(args.stream_step)])
        if args.stream_length:
            cmd.extend(["--stream-length", str(args.stream_length)])
        if args.stream_json:
            cmd.append("--stream-json")
        if args.auto_download:
            cmd.append("--auto-download")
        if args.crispasr_backend:
            cmd.extend(["--backend", args.crispasr_backend])
        elif backend_name.startswith("crispasr:"):
            cmd.extend(["--backend", backend_name.split(":", 1)[1]])

        logging.info(f"Starting stream: {' '.join(cmd)}")
        import subprocess

        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        proc.wait()
    finally:
        backend.cleanup()


def _warn_server_provenance(host, port):
    """Warn that synthetic audio served over HTTP is not marked by Susurrus.

    Reached only when the operator has taken the Art. 50 duty on with
    ``--accept-marking-responsibility``, which is the one way to run server
    mode without the marking proxy in front of it. Everything the endpoint
    emits is then marked by the binary alone, or not at all.
    """
    logging.warning("Server mode: AI-content marking is not verified by Susurrus.")
    print(
        f"NOTE: {host}:{port} is served by the CrispASR binary directly. Any "
        "synthetic audio it returns is marked by that binary alone — Susurrus "
        "is not in the response path and cannot verify EU AI Act Art. 50(2) "
        "marking or apply its declarative fallback. If the endpoint does TTS, "
        "verify a sample with 'susurrus --verify-c2pa FILE' before relying on "
        "it, and see COMPLIANCE.md.",
        file=sys.stderr,
    )


def _run_server(args):
    """Run CrispASR server mode behind the Art. 50 marking proxy.

    The binary is started on loopback and Susurrus binds the port the operator
    asked for, so every audio response passes through the same marking pipeline
    a local synthesis uses. Server mode used to hand the socket over directly,
    which made an HTTP endpoint the one route that emitted synthetic audio
    Susurrus never saw — and an endpoint reaches people who will never read a
    warning printed on the operator's terminal.

    If the proxy cannot be established the run is refused, unless the operator
    has taken the marking duty on. That is the same rule as everywhere else:
    either the software marked the output, or a named human said they would.
    """
    backend_name = args.backend
    model = args.model or "auto"

    if not backend_name.startswith("crispasr"):
        print("Error: server mode is only supported with crispasr backends", file=sys.stderr)
        sys.exit(1)

    # The proxy honours --no-watermark / --no-c2pa, so they reduce Art. 50
    # provenance here exactly as they do on the TTS path and need the same
    # attestation. Without this the one rule had an exception nobody declared.
    _require_marking_attestation(args)

    kwargs = _build_crispasr_kwargs(args)
    BackendClass = get_backend_class(backend_name)
    backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)

    host = args.host or "127.0.0.1"
    port = args.port or 8080
    attested = getattr(args, "accept_marking_responsibility", False)

    proxy = None
    proc = None
    try:
        if attested:
            # The operator has said the Art. 50 duty is theirs. Adding a proxy
            # they did not ask for would only slow their endpoint down.
            proc = backend.start_server(host=host, port=port)
            print(f"CrispASR server started on {host}:{port}")
            _warn_server_provenance(host, port)
            proc.wait()
            return

        proxy, proc = _start_marking_proxy(backend, host, port, args)
        print(f"CrispASR server started on {host}:{port} (behind the Art. 50 marking proxy)")
        print(
            "Audio responses are marked as AI-generated before they leave this "
            "process. Responses that cannot be marked are refused with a 502 "
            "rather than served unmarked.",
            file=sys.stderr,
        )
        proc.wait()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        if proxy is not None:
            proxy.stop()
        if proc is not None and proc.poll() is None:
            proc.terminate()
        backend.cleanup()


def _start_marking_proxy(backend, host, port, args):
    """Start the binary on loopback and Susurrus's marking proxy in front.

    Returns ``(proxy, process)``. Exits 2 rather than falling back to an
    unproxied server: silently degrading to the behaviour the operator was
    trying to avoid is how a control becomes decorative.
    """
    from utils.marking_proxy import MarkingProxy, find_free_port, wait_for_upstream

    upstream_host = "127.0.0.1"
    try:
        upstream_port = find_free_port(upstream_host)
    except OSError as e:
        _refuse_unproxied_server(f"could not reserve a loopback port for the backend ({e})")

    proc = backend.start_server(host=upstream_host, port=upstream_port)

    def _still_waiting(elapsed):
        print(
            f"Waiting for the CrispASR backend to start ({elapsed:.0f}s) — "
            "a first run downloads the model.",
            file=sys.stderr,
        )

    if not wait_for_upstream(upstream_host, upstream_port, process=proc, on_wait=_still_waiting):
        proc.terminate()
        _refuse_unproxied_server(
            f"the CrispASR server did not come up on {upstream_host}:{upstream_port}"
        )

    options = {
        key: getattr(args, key, None)
        for key in ("no_watermark", "no_c2pa", "c2pa_cert", "c2pa_key")
        if getattr(args, key, None)
    }

    try:
        proxy = MarkingProxy(
            listen_host=host,
            listen_port=port,
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            options=options,
            model=args.model or "auto",
        ).start()
    except OSError as e:
        proc.terminate()
        _refuse_unproxied_server(f"could not bind {host}:{port} ({e})")

    return proxy, proc


def _refuse_unproxied_server(reason):
    """Refuse to serve when the marking proxy cannot be established."""
    print(f"Refused: {reason}.", file=sys.stderr)
    print(
        "Server mode serves synthetic audio, and without the marking proxy "
        "Susurrus is not in the response path — it can neither mark nor verify "
        "what the endpoint emits. Fix the condition above, or pass "
        "--accept-marking-responsibility to run unproxied and take the EU AI "
        "Act Art. 50 obligation on yourself.",
        file=sys.stderr,
    )
    sys.exit(2)


def _run_align(args):
    """Run standalone alignment mode (text + audio, no ASR)."""
    if not args.file:
        print("Error: --file is required for alignment", file=sys.stderr)
        sys.exit(1)
    if not args.text_file and not args.text:
        print("Error: --text-file or --text is required for alignment", file=sys.stderr)
        sys.exit(1)

    backend_name = args.backend
    model = args.model or "auto"

    if not backend_name.startswith("crispasr"):
        print("Error: alignment mode is only supported with crispasr backends", file=sys.stderr)
        sys.exit(1)

    kwargs = _build_crispasr_kwargs(args)
    kwargs["align_only"] = True

    BackendClass = get_backend_class(backend_name)
    backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)

    try:
        audio_path = backend.preprocess_audio(args.file)
        for start, end, text in backend.transcribe(audio_path):
            if start > 0 or end > 0:
                print(f"[{start:.2f} --> {end:.2f}]  {text}")
            else:
                print(text)
    finally:
        backend.cleanup()


if __name__ == "__main__":
    main()
