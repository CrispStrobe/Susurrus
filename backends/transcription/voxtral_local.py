#!/usr/bin/env python3
# voxtral_local.py - Local Voxtral model inference

import logging
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from typing import List, Tuple

# Try to import librosa, but make it optional
try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not available. Install with: pip install librosa")

# Check if Voxtral is available
try:
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    VOXTRAL_AVAILABLE = True
except ImportError:
    VOXTRAL_AVAILABLE = False
    VoxtralForConditionalGeneration = None
    AutoProcessor = None
    logging.warning(
        "Voxtral not available. Install with:\n"
        "pip uninstall transformers -y\n"
        "pip install git+https://github.com/huggingface/transformers.git\n"
        "pip install mistral-common[audio] soundfile"
    )


class VoxtralLocal:
    """Local Voxtral model inference"""

    REPO_ID = "mistralai/Voxtral-Mini-3B-2507"
    SUPPORTED_LANGUAGES = ["en", "fr", "es", "de", "it", "pt", "pl", "nl"]

    def __init__(self, model_id: str = None, device: str = None):
        """Initialize Voxtral model

        Args:
            model_id: Model ID (default: mistralai/Voxtral-Mini-3B-2507)
            device: Device to use (cuda, mps, cpu, or auto)
        """
        if not VOXTRAL_AVAILABLE:
            raise ImportError(
                "Voxtral is not available. Please install requirements:\n"
                "pip uninstall transformers -y\n"
                "pip install git+https://github.com/huggingface/transformers.git\n"
                "pip install mistral-common[audio] soundfile"
            )

        self.model_id = model_id or self.REPO_ID
        self.device = self._detect_device() if device == "auto" or device is None else device
        self.model = None
        self.processor = None

        logging.info(f"Voxtral will use device: {self.device}")

    def _detect_device(self):
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        """Load the Voxtral model and processor"""
        if self.model is not None:
            return

        logging.info(f"Loading Voxtral model: {self.model_id}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.REPO_ID)

        # Load model
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.REPO_ID, torch_dtype=torch.bfloat16, device_map=self.device
        )

        logging.info("Voxtral model loaded successfully")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            if HAS_LIBROSA:
                audio_data, sr = librosa.load(audio_path, sr=None)
                return len(audio_data) / sr
            else:
                # Fallback to pydub
                from pydub import AudioSegment

                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0
        except Exception as e:
            logging.warning(f"Could not determine audio duration: {e}")
            return 30.0

    def _segment_audio(
        self, audio_path: str, chunk_length: int = 1500, chunk_overlap: int = 2
    ) -> List[Tuple[float, float, str]]:
        """
        Segment audio into chunks for processing long files.

        Args:
            audio_path: Path to audio file
            chunk_length: Length of each chunk in seconds (default: 1500 = 25 minutes)
            chunk_overlap: Overlap between chunks in seconds (default: 2)

        Returns:
            List of (start_time, end_time, chunk_audio_path) tuples
        """
        if HAS_LIBROSA:
            # Use librosa if available
            audio_data, sr = librosa.load(audio_path, sr=None)
        else:
            # Fallback to pydub + soundfile
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
            sr = audio.frame_rate
            # Convert to numpy array
            audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
            audio_data = audio_data / (2**15)  # Normalize to -1 to 1
            if audio.channels == 2:
                audio_data = audio_data.reshape((-1, 2)).mean(axis=1)

        audio_duration = len(audio_data) / sr

        chunks = []
        chunk_start = 0.0

        while chunk_start < audio_duration:
            chunk_end = min(chunk_start + chunk_length, audio_duration)

            # Extract chunk audio data
            start_sample = int(chunk_start * sr)
            end_sample = int(chunk_end * sr)
            chunk_data = audio_data[start_sample:end_sample]

            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                chunk_path = temp_file.name
                sf.write(chunk_path, chunk_data, sr)

            chunks.append((chunk_start, chunk_end, chunk_path))

            # Move to next chunk with overlap
            chunk_start = chunk_end - chunk_overlap

            # Don't create tiny chunks at the end
            if chunk_start >= audio_duration - chunk_overlap:
                break

        return chunks

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        max_new_tokens: int = 32000,
        temperature: float = 0.0,
        chunk_length: int = 1500,
    ) -> List[dict]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (en, fr, es, de, it, pt, pl, nl) or None for auto-detect
            max_new_tokens: Maximum tokens to generate per chunk
            temperature: Sampling temperature (0.0 = greedy)
            chunk_length: Maximum chunk length in seconds for long files

        Returns:
            List of segments with text, start, end times
        """
        # Ensure model is loaded
        self.load_model()

        # Get audio duration
        audio_duration = self._get_audio_duration(audio_path)

        # Determine if we need to segment
        if audio_duration > chunk_length:
            logging.info(f"Audio is {audio_duration:.1f}s, segmenting into {chunk_length}s chunks")
            return self._transcribe_long_audio(
                audio_path, language, max_new_tokens, temperature, chunk_length
            )
        else:
            return self._transcribe_single(
                audio_path, language, max_new_tokens, temperature, audio_duration
            )

    def _transcribe_single(
        self,
        audio_path: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        audio_duration: float,
    ) -> List[dict]:
        """Transcribe a single audio file (< 25 minutes)"""

        # Auto-detect language if not specified
        if language is None:
            language = "en"
            logging.info(f"No language specified, using default: {language}")
        elif language not in self.SUPPORTED_LANGUAGES:
            logging.warning(f"Language '{language}' not in supported list, using anyway")

        # Apply transcription request
        inputs = self.processor.apply_transcription_request(
            language=language, audio=audio_path, model_id=self.REPO_ID
        )
        inputs = inputs.to(self.device, dtype=torch.bfloat16)

        # Generate transcription
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                do_sample=temperature > 0,
            )

        # Decode outputs
        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        transcription_text = decoded_outputs[0].strip() if decoded_outputs else ""

        return [{"text": transcription_text, "start": 0.0, "end": audio_duration}]

    def _transcribe_long_audio(
        self,
        audio_path: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        chunk_length: int,
    ) -> List[dict]:
        """Transcribe long audio by segmenting it"""

        # Segment the audio
        audio_chunks = self._segment_audio(audio_path, chunk_length, chunk_overlap=2)
        logging.info(f"Split audio into {len(audio_chunks)} chunks")

        segments_result = []
        temp_files_to_cleanup = []

        try:
            for i, (chunk_start, chunk_end, chunk_path) in enumerate(audio_chunks):
                temp_files_to_cleanup.append(chunk_path)

                logging.info(
                    f"Processing chunk {i+1}/{len(audio_chunks)} "
                    f"({chunk_start:.0f}s - {chunk_end:.0f}s)"
                )

                # Determine language for first chunk or use provided
                lang = language if language else "en"

                # Apply transcription request
                inputs = self.processor.apply_transcription_request(
                    language=lang, audio=chunk_path, model_id=self.REPO_ID
                )
                inputs = inputs.to(self.device, dtype=torch.bfloat16)

                # Generate transcription
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 0.0,
                        do_sample=temperature > 0,
                    )

                # Decode outputs
                decoded_outputs = self.processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                transcription_text = decoded_outputs[0].strip() if decoded_outputs else ""

                if transcription_text:
                    segments_result.append(
                        {"text": transcription_text, "start": chunk_start, "end": chunk_end}
                    )

                # Free GPU memory between chunks
                if self.device in ["cuda", "mps"]:
                    torch.cuda.empty_cache() if self.device == "cuda" else None

        finally:
            # Clean up temporary chunk files
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logging.warning(f"Could not remove temp file {temp_file}: {e}")

        return segments_result

    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logging.info("Voxtral model unloaded")
