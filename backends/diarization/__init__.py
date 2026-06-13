"""Diarization backend - automatically initializes compatibility fixes"""

import logging

# Configure logging only if the application/root logger has not been set up
# yet. A library module must not unconditionally attach its own StreamHandler
# at import time — that hijacks the host application's logging and can bind to a
# transient stream (e.g. pytest's capture buffer).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

# Import and run compatibility fixes FIRST. These imports are intentionally
# not at the top of the file: the compatibility shims must be applied (and the
# device initialized) before pyannote.audio is imported via .manager below.
from .compat import initialize_diarization_backend  # noqa: E402

# Initialize on import (sets the module-level device / einsum flags)
default_device, has_opt_einsum = initialize_diarization_backend()

# Now import the main classes (they'll use the initialized environment). These
# pull in heavy optional dependencies (pyannote.audio, etc.); if they are not
# installed we degrade gracefully — the compatibility shims and device info
# above remain usable, and DIARIZATION_AVAILABLE signals that the actual
# diarization API is not. This keeps `import backends.diarization` working on
# environments that only have torch/torchaudio (e.g. CI, or transcription-only
# installs) instead of crashing the whole import.
try:
    from .manager import (  # noqa: E402
        DiarizationManager,
        time_to_srt,
        time_to_vtt,
        verify_authentication,
    )
    from .progress import (  # noqa: E402
        EnhancedProgress,
        optimize_pipeline,
        process_with_progress,
    )

    DIARIZATION_AVAILABLE = True
except ImportError as _import_error:
    logging.warning(
        "Diarization API unavailable — optional dependencies are missing: %s",
        _import_error,
    )
    DiarizationManager = None
    verify_authentication = None
    EnhancedProgress = None
    optimize_pipeline = None
    process_with_progress = None
    time_to_srt = None
    time_to_vtt = None
    DIARIZATION_AVAILABLE = False

__all__ = [
    "DiarizationManager",
    "verify_authentication",
    "EnhancedProgress",
    "process_with_progress",
    "optimize_pipeline",
    "time_to_srt",
    "time_to_vtt",
    "default_device",
    "has_opt_einsum",
    "DIARIZATION_AVAILABLE",
]
