#!/usr/bin/env python3
# voxtral_api.py - Voxtral API inference via Mistral AI API

import logging
import os
import tempfile
import time

import requests
from typing import List, Dict

class VoxtralAPI:
    """Voxtral inference via Mistral AI API"""

    API_BASE = "https://api.mistral.ai/v1"
    AUDIO_ENDPOINT = "/audio/transcriptions"

    def __init__(self, api_key: str = None):
        """Initialize Voxtral API client

        Args:
            api_key: Mistral AI API key (or set MISTRAL_API_KEY env var)
        """
        # Try multiple ways to get the API key
        if api_key:
            self.api_key = api_key
        else:
            # Try environment variable
            self.api_key = os.environ.get("MISTRAL_API_KEY")

            # If still not found, try reading from a config file
            if not self.api_key:
                config_file = os.path.expanduser("~/.mistral/api_key")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, "r") as f:
                            self.api_key = f.read().strip()
                    except Exception:
                        pass

        if not self.api_key:
            raise ValueError(
                "Mistral API key required. Provide via:\n"
                "  1. --mistral-api-key argument\n"
                "  2. MISTRAL_API_KEY environment variable (PowerShell: $env:MISTRAL_API_KEY = 'key')\n"
                "  3. ~/.mistral/api_key file\n"
                "\nGet your API key from: https://console.mistral.ai/"
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
            }
        )

    def transcribe(
        self, audio_path: str, language: str = None, model: str = "voxtral-mini-latest"
    ) -> List[Dict]:
        """
        Transcribe audio file using Mistral API.
        """
        logging.info(f"Transcribing via Mistral API: {audio_path}")

        # Prepare the audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Get file size for logging
        file_size_mb = len(audio_data) / (1024 * 1024)
        logging.info(f"Audio file size: {file_size_mb:.2f} MB")

        # Prepare the request
        files = {"file": (os.path.basename(audio_path), audio_data, "audio/mpeg")}

        data = {"model": model, "response_format": "verbose_json"}  # Get timestamps

        if language:
            data["language"] = language
            logging.info(f"Language set to: {language}")

        # Make the API request
        url = f"{self.API_BASE}{self.AUDIO_ENDPOINT}"

        try:
            logging.info(f"Sending request to: {url}")
            response = self.session.post(url, files=files, data=data, timeout=300)
            response.raise_for_status()

            result = response.json()

            # DEBUG: Print full response structure
            logging.info(f"Full API response: {result}")

            # Parse response - prioritize segments if they exist and are not empty
            segments = []

            # Check if segments exist and are not empty
            if "segments" in result and result["segments"]:
                logging.info(f"Found {len(result['segments'])} segments in response")
                for segment in result["segments"]:
                    seg_data = {
                        "text": segment.get("text", ""),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                    }
                    segments.append(seg_data)
            elif "text" in result and result["text"]:
                # Response contains full text without segments
                text = result["text"].strip()
                logging.info(f"Response contains full text ({len(text)} chars) without segments")
                segments.append({"text": text, "start": 0.0, "end": 0.0})
            else:
                logging.warning(f"Unexpected API response format. Keys: {list(result.keys())}")
                # Try to extract text from any available field
                text = str(result.get("transcription", result.get("transcript", "")))
                if text:
                    segments.append({"text": text, "start": 0.0, "end": 0.0})

            logging.info(f"Returning {len(segments)} segments")
            return segments

        except requests.exceptions.HTTPError as e:
            error_text = e.response.text

            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your Mistral API key.\n"
                    "Get your API key from: https://console.mistral.ai/"
                )
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            else:
                raise ValueError(f"API request failed: {error_text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error during API request: {str(e)}")

    def transcribe_with_chunking(
        self, audio_path: str, language: str = None, max_duration: int = 600
    ) -> List[dict]:
        """
        Transcribe long audio by splitting it into chunks.
        Uses pydub instead of librosa to avoid numba dependency.

        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            max_duration: Maximum duration per chunk in seconds (default: 10 minutes)

        Returns:
            List of segments with adjusted timestamps
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            logging.error("pydub is required for audio chunking. Install with: pip install pydub")
            # Fall back to direct transcription without chunking
            return self.transcribe(audio_path, language)

        # Load audio
        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # pydub uses milliseconds
        except Exception as e:
            logging.warning(f"Could not load audio with pydub: {e}. Trying direct transcription.")
            return self.transcribe(audio_path, language)

        if duration <= max_duration:
            # Audio is short enough, transcribe directly
            logging.info(f"Audio duration {duration:.1f}s is within limit, transcribing directly")
            return self.transcribe(audio_path, language)

        # Split audio into chunks
        logging.info(f"Audio is {duration:.1f}s, splitting into {max_duration}s chunks")

        chunks = []
        chunk_start = 0.0
        max_duration_ms = max_duration * 1000  # Convert to milliseconds

        while chunk_start < duration:
            chunk_end = min(chunk_start + max_duration, duration)

            start_ms = int(chunk_start * 1000)
            end_ms = int(chunk_end * 1000)

            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]

            # Save chunk to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                chunk_path = f.name
                chunk_audio.export(chunk_path, format="wav")

            chunks.append((chunk_start, chunk_end, chunk_path))
            chunk_start = chunk_end

        # Transcribe each chunk
        all_segments = []

        try:
            for i, (chunk_start, chunk_end, chunk_path) in enumerate(chunks):
                logging.info(
                    f"Transcribing chunk {i+1}/{len(chunks)} "
                    f"({chunk_start:.0f}s - {chunk_end:.0f}s)"
                )

                segments = self.transcribe(chunk_path, language)

                # Adjust timestamps
                for segment in segments:
                    if segment["start"] is not None:
                        segment["start"] += chunk_start
                    if segment["end"] is not None:
                        segment["end"] += chunk_start
                    all_segments.append(segment)

                # Rate limiting - wait between chunks
                if i < len(chunks) - 1:
                    time.sleep(1)

        finally:
            # Clean up temp files
            for _, _, chunk_path in chunks:
                if os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass

        return all_segments
