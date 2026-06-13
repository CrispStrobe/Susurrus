"""Diarization backend - automatically initializes compatibility fixes"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# Import and run compatibility fixes FIRST. These imports are intentionally
# not at the top of the file: the compatibility shims must be applied (and the
# device initialized) before pyannote.audio is imported via .manager below.
from .compat import initialize_diarization_backend  # noqa: E402

# Initialize on import (sets the module-level device / einsum flags)
default_device, has_opt_einsum = initialize_diarization_backend()

# Now import the main classes (they'll use the initialized environment)
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
]
