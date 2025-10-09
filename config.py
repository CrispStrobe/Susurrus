"""Configuration and constants"""

import logging
import os
import platform

from PyQt6.QtCore import QSettings

APP_NAME = "Susurrus"
APP_VERSION = "1.1.0"
APP_ORG = "CrispStrobe"

BACKEND_MODEL_MAP = {
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
    "voxtral-local": [
        ("mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Mini-3B-2507"),
    ],
    "voxtral-api": [
        ("voxtral-mini-latest", "voxtral-mini-latest"),
    ],
}

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
}

# Available diarization models
DIARIZATION_MODELS = ["Default", "Legacy", "English", "Chinese", "German", "Spanish", "Japanese"]


def get_settings():
    """Get QSettings instance"""
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
            return "whisper.cpp"


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
    }
    return defaults.get(backend, "tiny")


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
