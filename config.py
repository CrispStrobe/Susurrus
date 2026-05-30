"""Configuration and constants"""

import os
import platform

APP_NAME = "Susurrus"
APP_VERSION = "2.0.1"
APP_ORG = "CrispStrobe"

# ---------------------------------------------------------------------------
# Transcription backend → model map
# Each entry: (model_id_or_path, canonical_display_name)
# For CrispASR sub-backends, "auto" triggers auto-download from the registry.
# ---------------------------------------------------------------------------
BACKEND_MODEL_MAP = {
    # --- Whisper-family backends ---
    "mlx-whisper": [
        ("mlx-community/whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"),
        ("mlx-community/whisper-large-v3-turbo-q4", "openai/whisper-large-v3-turbo"),
        ("mlx-community/whisper-tiny-mlx-4bit", "openai/whisper-tiny"),
        ("mlx-community/whisper-base-mlx-4bit", "openai/whisper-base"),
        ("mlx-community/whisper-small-mlx-q4", "openai/whisper-small"),
        ("mlx-community/whisper-medium-mlx-4bit", "openai/whisper-medium"),
        ("mlx-community/whisper-large-v3-mlx-4bit", "openai/whisper-large-v3"),
        ("mlx-community/whisper-large-v3-mlx", "openai/whisper-large-v3"),
    ],
    "faster-batched": [
        ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
        ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3-turbo"),
        ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
        ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3"),
    ],
    "faster-sequenced": [
        ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
        ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3-turbo"),
        ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
        ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3"),
    ],
    "transformers": [
        ("openai/whisper-large-v3", "openai/whisper-large-v3"),
        ("openai/whisper-large-v2", "openai/whisper-large-v2"),
        ("openai/whisper-medium", "openai/whisper-medium"),
        ("openai/whisper-small", "openai/whisper-small"),
    ],
    "OpenAI Whisper": [
        ("large-v2", "openai/whisper-large-v2"),
        ("medium", "openai/whisper-medium"),
        ("small", "openai/whisper-small"),
        ("base", "openai/whisper-base"),
        ("tiny", "openai/whisper-tiny"),
    ],
    "whisper.cpp": [
        ("large-v3-turbo-q5_0", "openai/whisper-large-v3"),
        ("large-v3-turbo", "openai/whisper-large-v3"),
        ("small", "openai/whisper-small"),
        ("large-v3-q5_0", "openai/whisper-large-v3"),
        ("medium-q5_0", "openai/whisper-medium"),
        ("small-q5_1", "openai/whisper-small"),
        ("base", "openai/whisper-base"),
        ("tiny", "openai/whisper-tiny"),
        ("tiny-q5_1", "openai/whisper-tiny"),
        ("tiny.en", "openai/whisper-tiny.en"),
    ],
    "ctranslate2": [
        ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
        ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3"),
        ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
        ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3"),
    ],
    "whisper-jax": [
        ("openai/whisper-tiny", "openai/whisper-tiny"),
        ("openai/whisper-medium", "openai/whisper-medium"),
        ("tiny.en", "openai/whisper-tiny.en"),
        ("base.en", "openai/whisper-base.en"),
        ("small.en", "openai/whisper-small.en"),
        ("medium.en", "openai/whisper-medium.en"),
        ("large-v2", "openai/whisper-large-v2"),
    ],
    "insanely-fast-whisper": [
        ("openai/whisper-large-v3", "openai/whisper-large-v3"),
        ("openai/whisper-medium", "openai/whisper-medium"),
        ("openai/whisper-small", "openai/whisper-small"),
        ("openai/whisper-base", "openai/whisper-base"),
        ("openai/whisper-tiny", "openai/whisper-tiny"),
    ],
    # --- Voxtral ---
    "voxtral-local": [
        ("mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Mini-3B-2507"),
    ],
    "voxtral-api": [
        ("voxtral-mini-latest", "voxtral-mini-latest"),
    ],
    # --- CrispASR: unified GGML/GGUF multi-backend engine ---
    # "auto" triggers auto-download; "auto:q8_0" specifies preferred quant
    "crispasr": [
        ("auto", "Auto-detect (default whisper model)"),
    ],
    # CrispASR FFI (in-process via Python bindings — faster, no subprocess)
    "crispasr-ffi": [
        ("auto", "Auto-detect (FFI — requires libcrispasr)"),
    ],
    # CrispASR ASR sub-backends
    "crispasr:whisper": [
        ("auto", "Whisper (auto-download)"),
        ("auto:q5_0", "Whisper Q5_0"),
    ],
    "crispasr:parakeet": [
        ("auto", "NVIDIA Parakeet TDT 0.6B"),
        ("auto:q8_0", "Parakeet Q8_0"),
    ],
    "crispasr:canary": [
        ("auto", "NVIDIA Canary 1B"),
        ("auto:q8_0", "Canary Q8_0"),
    ],
    "crispasr:cohere": [
        ("auto", "Cohere ASR"),
        ("auto:q4_0", "Cohere Q4_0"),
    ],
    "crispasr:qwen3": [
        ("auto", "Qwen3 Audio"),
        ("auto:q4_0", "Qwen3 Q4_0"),
    ],
    "crispasr:voxtral": [
        ("auto", "Voxtral Mini 3B GGUF"),
        ("auto:q4_0", "Voxtral Q4_0"),
    ],
    "crispasr:voxtral4b": [
        ("auto", "Voxtral 4B"),
        ("auto:q4_0", "Voxtral4B Q4_0"),
    ],
    "crispasr:granite": [
        ("auto", "IBM Granite Speech 3.3 8B"),
        ("auto:q4_0", "Granite Q4_0"),
    ],
    "crispasr:moonshine": [
        ("auto", "Moonshine Base"),
        ("auto:q8_0", "Moonshine Q8_0"),
    ],
    "crispasr:kyutai-stt": [
        ("auto", "Kyutai Moshi STT"),
        ("auto:q8_0", "Kyutai Q8_0"),
    ],
    "crispasr:fastconformer-ctc": [
        ("auto", "FastConformer CTC"),
        ("auto:q8_0", "FastConformer Q8_0"),
    ],
    "crispasr:wav2vec2": [
        ("auto", "Wav2Vec2 Base"),
        ("auto:q8_0", "Wav2Vec2 Q8_0"),
    ],
    "crispasr:firered-asr": [
        ("auto", "FireRed ASR"),
        ("auto:q8_0", "FireRed Q8_0"),
    ],
    "crispasr:funasr": [
        ("auto", "FunASR"),
        ("auto:q8_0", "FunASR Q8_0"),
    ],
    "crispasr:glm-asr": [
        ("auto", "GLM ASR"),
        ("auto:q4_0", "GLM Q4_0"),
    ],
    "crispasr:omniasr": [
        ("auto", "OmniASR"),
        ("auto:q8_0", "OmniASR Q8_0"),
    ],
    "crispasr:vibevoice-asr": [
        ("auto", "VibeVoice ASR"),
        ("auto:q4_0", "VibeVoice Q4_0"),
    ],
    "crispasr:gemma4-e2b": [
        ("auto", "Gemma4 E2B (ASR+MT)"),
        ("auto:q4_0", "Gemma4 Q4_0"),
    ],
    # CrispASR Translation backends
    "crispasr:m2m100": [
        ("auto", "M2M100 Translation"),
        ("auto:q4_0", "M2M100 Q4_0"),
    ],
    "crispasr:madlad": [
        ("auto", "MadLad 419-language Translation"),
        ("auto:q4_0", "MadLad Q4_0"),
    ],
}

# ---------------------------------------------------------------------------
# TTS backend configuration
# ---------------------------------------------------------------------------
TTS_BACKEND_MAP = {
    # CrispASR-based TTS backends
    "crispasr:kokoro": {
        "models": [("auto", "Kokoro TTS"), ("auto:q8_0", "Kokoro Q8_0")],
        "voices": ["af_heart", "af_sky", "am_adam", "bf_emma", "bm_george"],
        "default_voice": "af_heart",
    },
    "crispasr:orpheus": {
        "models": [("auto", "Orpheus TTS"), ("auto:q4_0", "Orpheus Q4_0")],
        "voices": ["Anton", "Sophie", "Tara", "Leo", "Leah"],
        "default_voice": "Tara",
    },
    "crispasr:qwen3-tts": {
        "models": [("auto", "Qwen3 TTS"), ("auto:q4_0", "Qwen3 TTS Q4_0")],
        "voices": [],
        "default_voice": None,
    },
    "crispasr:chatterbox-tts": {
        "models": [("auto", "Chatterbox TTS"), ("auto:q8_0", "Chatterbox Q8_0")],
        "voices": [],
        "default_voice": None,
    },
    "crispasr:vibevoice-tts": {
        "models": [("auto", "VibeVoice TTS")],
        "voices": [],
        "default_voice": None,
    },
    "crispasr:indextts": {
        "models": [("auto", "IndexTTS")],
        "voices": [],
        "default_voice": None,
    },
    "crispasr:voxcpm2-tts": {
        "models": [("auto", "VoxCPM2 TTS 48kHz")],
        "voices": [],
        "default_voice": None,
    },
    # Python-native TTS backends (from CrispTTS)
    "edge-tts": {
        "models": [("edge-tts", "Microsoft Edge TTS (cloud)")],
        "voices": [
            "de-DE-KatjaNeural",
            "de-DE-ConradNeural",
            "de-DE-AmalaNeural",
            "de-DE-BerndNeural",
            "de-DE-ChristophNeural",
            "de-DE-ElkeNeural",
            "de-DE-GiselaNeural",
            "de-DE-KasperNeural",
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-GB-SoniaNeural",
            "en-GB-RyanNeural",
            "fr-FR-DeniseNeural",
            "fr-FR-HenriNeural",
            "es-ES-ElviraNeural",
            "es-ES-AlvaroNeural",
        ],
        "default_voice": "de-DE-KatjaNeural",
    },
    "piper": {
        "models": [("piper", "Piper ONNX TTS (local)")],
        "voices": [
            "de_DE-thorsten-medium",
            "de_DE-thorsten-high",
            "de_DE-thorsten-low",
            "de_DE-eva_k-x_low",
            "de_DE-karlsson-low",
            "de_DE-kerstin-low",
            "de_DE-pavoque-low",
            "de_DE-ramona-low",
            "en_US-lessac-medium",
            "en_US-amy-medium",
            "en_GB-alba-medium",
        ],
        "default_voice": "de_DE-thorsten-medium",
    },
    "kokoro-onnx": {
        "models": [("kokoro-onnx", "Kokoro ONNX TTS (local)")],
        "voices": [
            "af_heart",
            "af_sky",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ],
        "default_voice": "af_heart",
    },
    "chatterbox": {
        "models": [("chatterbox", "Chatterbox TTS (local)")],
        "voices": [],
        "default_voice": None,
    },
    "speecht5": {
        "models": [
            ("microsoft/speecht5_tts", "SpeechT5 TTS"),
        ],
        "voices": [],
        "default_voice": None,
    },
}

# All CrispASR sub-backend names for listing
CRISPASR_SUB_BACKENDS = [
    "whisper",
    "parakeet",
    "canary",
    "cohere",
    "qwen3",
    "voxtral",
    "voxtral4b",
    "granite",
    "moonshine",
    "kyutai-stt",
    "fastconformer-ctc",
    "wav2vec2",
    "firered-asr",
    "funasr",
    "glm-asr",
    "omniasr",
    "vibevoice-asr",
    "gemma4-e2b",
]

# Companion models required by certain TTS backends (auto-download)
# Maps backend → list of (companion_role, registry_name)
CRISPASR_COMPANION_MODELS = {
    "qwen3-tts": [("codec", "qwen3-tts-tokenizer-12hz")],
    "orpheus": [("codec", "orpheus-snac-codec")],
    "mimo-asr": [("codec", "mimo-tokenizer")],
    "vibevoice-tts": [("voice", "vibevoice-default-voice")],
}

CRISPASR_TTS_BACKENDS = [
    "kokoro",
    "orpheus",
    "qwen3-tts",
    "chatterbox-tts",
    "vibevoice-tts",
    "indextts",
    "voxcpm2-tts",
]

CRISPASR_TRANSLATION_BACKENDS = [
    "m2m100",
    "madlad",
    "gemma4-e2b",
]

# Device fallbacks for backends
DEVICE_FALLBACKS = {
    "faster-batched": [("mps", "cpu")],
    "faster-sequenced": [("mps", "cpu")],
    "faster-whisper": [("mps", "cpu")],
    "openai-whisper": [("mps", "cpu")],
    "whisper-jax": [("mps", "cpu")],
    "ctranslate2": [("mps", "cpu"), ("cuda", "cpu")],
    "voxtral-local": [],
    "voxtral-api": [],
    "crispasr": [],
    "crispasr-ffi": [],
}

# Available diarization models (PyAnnote-based, for Susurrus's own diarization)
DIARIZATION_MODELS = ["Default", "Legacy", "English", "Chinese", "German", "Spanish", "Japanese"]

# CrispASR diarization methods (via the binary)
CRISPASR_DIARIZE_METHODS = [
    "energy",
    "xcorr",
    "vad-turns",
    "pyannote",
    "sherpa",
    "ecapa",
]

# CrispASR LID methods
CRISPASR_LID_BACKENDS = ["whisper", "silero", "firered", "ecapa"]

# Supported languages for various features
SUPPORTED_LANGUAGES = {
    "voxtral": ["en", "fr", "es", "de", "it", "pt", "pl", "nl"],
    "crispasr": [
        "auto",
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
    ],
}


def get_settings():
    """Get QSettings instance (only available when PyQt6 is installed)."""
    from PyQt6.QtCore import QSettings

    return QSettings(APP_ORG, APP_NAME)


def get_default_backend():
    """Get default backend based on platform"""
    system = platform.system().lower()

    if system == "darwin":
        try:
            return "mlx-whisper"
        except ImportError:
            return "faster-batched"
    elif system == "windows":
        from utils.device_detection import check_cuda

        if check_cuda():
            return "faster-batched"
        else:
            return "whisper.cpp"
    else:  # Linux
        from utils.device_detection import check_cuda

        if check_cuda():
            return "faster-batched"
        else:
            return "crispasr"


def get_default_model_for_backend(backend):
    """Get default model for a backend"""
    defaults = {
        "mlx-whisper": "mlx-community/whisper-tiny-mlx-4bit",
        "faster-batched": "cstr/whisper-large-v3-turbo-int8_float32",
        "faster-sequenced": "cstr/whisper-large-v3-turbo-int8_float32",
        "whisper.cpp": "tiny",
        "transformers": "openai/whisper-tiny",
        "OpenAI Whisper": "tiny",
        "ctranslate2": "cstr/whisper-large-v3-turbo-int8_float32",
        "whisper-jax": "openai/whisper-tiny",
        "insanely-fast-whisper": "openai/whisper-tiny",
        "voxtral-local": "mistralai/Voxtral-Mini-3B-2507",
        "voxtral-api": "voxtral-mini-latest",
        "crispasr": "auto",
        "crispasr-ffi": "auto",
    }
    # Handle crispasr/crispasr-ffi sub-backends
    if backend.startswith("crispasr:") or backend.startswith("crispasr-ffi:"):
        return "auto"
    return defaults.get(backend, "auto")


def get_supported_device(backend, device):
    """Get supported device for backend with fallbacks"""
    if backend in DEVICE_FALLBACKS:
        for unsupported, fallback in DEVICE_FALLBACKS[backend]:
            if device.lower() == unsupported.lower():
                import logging

                logging.warning(
                    f"Device '{device}' not supported by '{backend}'. Using '{fallback}'."
                )
                return fallback
    return device


def get_resource_path(filename):
    """Get path to resource file"""
    return os.path.join(os.path.dirname(__file__), "resources", filename)
