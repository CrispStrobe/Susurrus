#!/usr/bin/env python3
"""
diarize_audio.py - Standalone CLI script for speaker diarization

This script can be run independently or imported as a module.
When imported, it ensures all compatibility fixes are applied.
"""

import logging
import os
import sys

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modularized backend
from backends.diarization import DiarizationManager, default_device, verify_authentication


def test_diarization(audio_path, hf_token=None, device=None, num_speakers=None):
    """Test the diarization functionality"""
    logging.info(f"Starting speaker diarization...")

    # Use the provided device or fall back to default
    if device is None:
        device = str(default_device).split(":")[0]

    manager = DiarizationManager(hf_token=hf_token, device=device)

    # First verify authentication directly
    token = manager._get_token()
    if token:
        logging.info(f"Using token with length: {len(token)} characters")
    else:
        logging.warning("No token found!")

    try:
        # If num_speakers is provided, use it directly
        if num_speakers is not None:
            segments, segment_files = manager.diarize_and_split(
                audio_path, max_speakers=num_speakers
            )
        else:
            segments, segment_files = manager.diarize_and_split(audio_path)

        logging.info(f"Diarization completed successfully on {audio_path}")
        logging.info(
            f"Found {len(set(s['speaker'] for s in segments))} speakers in {len(segments)} segments"
        )

        # Print the first few segments
        for i, segment in enumerate(segments[:5]):
            logging.info(
                f"{i}: {segment['speaker']} ({segment['start']:.2f}s - {segment['end']:.2f}s)"
            )

        return segments, segment_files
    except Exception as e:
        logging.error(f"Speaker diarization process failed: {e}")
        return None, None


if __name__ == "__main__":
    print("Audio diarization utility")
    if len(sys.argv) < 2:
        print(
            "Usage: python diarize_audio.py <audio_file> [huggingface_token] [device] [num_speakers]"
        )
        print("       python diarize_audio.py --verify-auth [huggingface_token]")
        sys.exit(1)

    if sys.argv[1] == "--verify-auth":
        token = sys.argv[2] if len(sys.argv) > 2 else None
        verify_authentication(token)
        sys.exit(0)

    audio_path = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None
    device = sys.argv[3] if len(sys.argv) > 3 else None
    num_speakers = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else None

    print(
        f"Parameters: audio {audio_path}, token: {'provided' if hf_token else 'not provided'}, device: {device}, num_speakers: {num_speakers}"
    )

    test_diarization(audio_path, hf_token, device, num_speakers)
