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
        choices=["transcribe", "tts", "translate", "stream", "server"],
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
    tts_group.add_argument("--tts-steps", type=int, default=None, help="TTS diffusion steps")
    tts_group.add_argument("--play", action="store_true", help="Play audio after synthesis")
    tts_group.add_argument(
        "--tts-play", action="store_true", help="Play audio on local speaker (CrispASR native)"
    )
    tts_group.add_argument(
        "--tts-play-device", type=int, default=None, help="Audio device index for local playback"
    )
    tts_group.add_argument("--list-voices", action="store_true", help="List voices for TTS backend")

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

    # --- CrispASR punctuation ---
    punc_group = parser.add_argument_group("CrispASR Punctuation Options")
    punc_group.add_argument("--punc-model", default=None, help="Punctuation restoration model")

    # --- CrispASR speaker ---
    spk_group = parser.add_argument_group("CrispASR Speaker Options")
    spk_group.add_argument("--speaker-db", default=None, help="Speaker profile database path")
    spk_group.add_argument("--enroll-speaker", default=None, help="Enroll speaker name")
    spk_group.add_argument("--speaker-threshold", type=float, default=None, help="Match threshold")
    spk_group.add_argument("--titanet-model", default=None, help="Speaker embedding model")

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

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if args.list_backends:
        _list_backends()
        return

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
        # Diarization
        "diarize": "diarize",
        "diarize_method": "diarize_method",
        "diarize_embedder": "diarize_embedder",
        "diarize_cluster_threshold": "diarize_cluster_threshold",
        "diarize_max_speakers": "diarize_max_speakers",
        # LID
        "detect_language": "detect_language",
        "lid_backend": "lid_backend",
        "lid_model": "lid_model",
        # Alignment
        "aligner_model": "aligner_model",
        "force_aligner": "force_aligner",
        # Punctuation
        "punc_model": "punc_model",
        # Speaker
        "speaker_db": "speaker_db",
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
        "tts_steps": "tts_steps",
        "tts_play": "tts_play",
        "tts_play_device": "tts_play_device",
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
        for start, end, text in backend.transcribe(audio_path):
            if start > 0 or end > 0:
                print(f"[{start:.2f} --> {end:.2f}]  {text}")
            else:
                print(text)
    finally:
        backend.cleanup()


def _run_tts(args):
    """Run TTS mode."""
    text = _read_input_text(args)
    if not text:
        print("Error: --text or --input-file is required for TTS", file=sys.stderr)
        sys.exit(1)

    tts_backend = args.tts_backend or args.backend
    output_path = args.tts_output

    # Route to CrispASR TTS or Python TTS backend
    if tts_backend.startswith("crispasr"):
        model = args.model or "auto"
        kwargs = _build_crispasr_kwargs(args)
        BackendClass = get_backend_class(tts_backend)
        backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)
        try:
            result = backend.synthesize(text, output_path)
            print(f"Audio saved to: {result}")
        finally:
            backend.cleanup()
    else:
        TTSClass = get_tts_backend_class(tts_backend)
        tts_kwargs = {}
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
            print(f"Audio saved to: {result}")
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
    finally:
        backend.cleanup()


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


def _run_server(args):
    """Run CrispASR server mode."""
    backend_name = args.backend
    model = args.model or "auto"

    if not backend_name.startswith("crispasr"):
        print("Error: server mode is only supported with crispasr backends", file=sys.stderr)
        sys.exit(1)

    kwargs = _build_crispasr_kwargs(args)
    BackendClass = get_backend_class(backend_name)
    backend = BackendClass(model_id=model, device=args.device, language=args.language, **kwargs)

    host = args.host or "127.0.0.1"
    port = args.port or 8080

    try:
        proc = backend.start_server(host=host, port=port)
        print(f"CrispASR server started on {host}:{port}")
        proc.wait()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        backend.cleanup()


if __name__ == "__main__":
    main()
