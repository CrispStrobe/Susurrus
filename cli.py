#!/usr/bin/env python3
"""Susurrus CLI — headless transcription using any backend.

Usage:
    python cli.py --backend crispasr --model parakeet.gguf --file audio.wav
    python cli.py --backend faster-sequenced --model large-v3 --file audio.wav
    python cli.py --list-backends

Backends: mlx-whisper, faster-batched, faster-sequenced, transformers,
          whisper.cpp, ctranslate2, whisper-jax, insanely-fast-whisper,
          openai whisper, voxtral-local, voxtral-api, crispasr
"""

import argparse
import logging
import sys
import os

# Minimal imports — avoid PyQt6
sys.path.insert(0, os.path.dirname(__file__))


def get_backend_class(name):
    """Import backend class without triggering PyQt6."""
    backends_dir = os.path.join(os.path.dirname(__file__),
                                "workers", "transcription", "backends")
    sys.path.insert(0, backends_dir)

    # Create fake package modules to satisfy relative imports
    import types
    for mod_name in [
        "workers", "workers.transcription", "workers.transcription.backends",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # Load base
    import importlib.util
    base_spec = importlib.util.spec_from_file_location(
        "workers.transcription.backends.base",
        os.path.join(backends_dir, "base.py"))
    base_mod = importlib.util.module_from_spec(base_spec)
    base_mod.__package__ = "workers.transcription.backends"
    sys.modules["workers.transcription.backends.base"] = base_mod
    base_spec.loader.exec_module(base_mod)
    sys.modules["workers.transcription.backends"].TranscriptionBackend = \
        base_mod.TranscriptionBackend

    # Map name → file
    backend_files = {
        "crispasr": "crispasr_backend.py",
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

    if name not in backend_files:
        raise ValueError(f"Unknown backend: {name}. Available: {', '.join(sorted(backend_files))}")

    fname = backend_files[name]
    cname = backend_classes[name]

    spec = importlib.util.spec_from_file_location(
        f"workers.transcription.backends.{fname[:-3]}",
        os.path.join(backends_dir, fname))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "workers.transcription.backends"
    spec.loader.exec_module(mod)
    return getattr(mod, cname)


def main():
    parser = argparse.ArgumentParser(
        description="Susurrus CLI — headless transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--backend", "-b", default="crispasr",
                        help="Transcription backend (default: crispasr)")
    parser.add_argument("--model", "-m", default=None,
                        help="Model path or HF model ID")
    parser.add_argument("--file", "-f", default=None,
                        help="Audio file to transcribe")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (e.g. en, de)")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Device (cpu, cuda, mps)")
    parser.add_argument("--list-backends", action="store_true",
                        help="List available backends and exit")

    # CrispASR-specific
    parser.add_argument("--crispasr-backend", default=None,
                        help="Force CrispASR sub-backend (parakeet, qwen3, ...)")
    parser.add_argument("--vad", action="store_true",
                        help="Enable VAD (CrispASR)")
    parser.add_argument("--split-on-punct", action="store_true",
                        help="Split at punctuation (CrispASR)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list_backends:
        backends = [
            "crispasr", "faster-batched", "faster-sequenced", "transformers",
            "whisper.cpp", "ctranslate2", "whisper-jax", "insanely-fast-whisper",
            "openai whisper", "voxtral-local", "voxtral-api", "mlx-whisper",
        ]
        for b in backends:
            print(f"  {b}")
        return

    if not args.model or not args.file:
        parser.error("--model and --file are required for transcription")

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    BackendClass = get_backend_class(args.backend)

    kwargs = {}
    if args.crispasr_backend:
        kwargs["crispasr_backend"] = args.crispasr_backend
    if args.vad:
        kwargs["vad"] = True
    if args.split_on_punct:
        kwargs["split_on_punct"] = True

    backend = BackendClass(
        model_id=args.model,
        device=args.device,
        language=args.language,
        **kwargs)

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
