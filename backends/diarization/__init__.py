"""Diarization backend - automatically initializes compatibility fixes"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# Import and run compatibility fixes FIRST
from .compat import default_device, has_opt_einsum, initialize_diarization_backend

# Initialize on import
default_device, has_opt_einsum = initialize_diarization_backend()

# Now import the main classes (they'll use the initialized environment)
from .manager import DiarizationManager, time_to_srt, time_to_vtt, verify_authentication
from .progress import EnhancedProgress, optimize_pipeline, process_with_progress

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
