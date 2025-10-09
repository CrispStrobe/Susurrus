# backends/diarization/manager.py
"""Diarization manager"""
import logging
import os
import tempfile

import torch
from huggingface_hub import HfApi, hf_hub_download
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Import the default device from compat module
from .compat import default_device


class DiarizationManager:
    """Manager class for speaker diarization using pyannote.audio"""

    AVAILABLE_MODELS = {
        "Default": "pyannote/speaker-diarization-3.1",
        "Legacy": "pyannote/speaker-diarization@2.1",
    }

    def __init__(self, hf_token=None, device=None, model_name="Default"):
        """Initialize diarization manager

        Args:
            hf_token: Hugging Face API token
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
            model_name: Name of the diarization model to use
        """
        self.hf_token = hf_token
        self.model_name = model_name
        if model_name not in self.AVAILABLE_MODELS:
            logging.warning(f"Unknown model name: {model_name}. Falling back to Default model.")
            self.model_name = "Default"

        # Use the initialized default_device if none specified
        if device is None:
            self.device = str(default_device).split(":")[0]  # 'cuda', 'mps', or 'cpu'
        else:
            self.device = self._validate_device(device)

        self.pipeline = None
        logging.info(f"Diarization will use device: {self.device}")
        logging.info(f"Selected diarization model: {self.model_name}")

    def _validate_device(self, device):
        """Validate and adjust device if necessary"""
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        elif device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            logging.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    def _detect_device(self):
        """Auto-detect the best available device"""
        # This is now redundant with the global default_device, but kept for compatibility
        return str(default_device).split(":")[0]

    def get_model_id(self):
        """Get the model ID for the selected model name"""
        return self.AVAILABLE_MODELS.get(self.model_name, self.AVAILABLE_MODELS["Default"])

    @classmethod
    def list_available_models(cls):
        """Return a list of available models"""
        return list(cls.AVAILABLE_MODELS.keys())

    def _get_token(self):
        """Get the Hugging Face token from various sources"""
        if self.hf_token is not None:
            return self.hf_token

        # Try environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            self.hf_token = token
            return token

        # Try token file
        try:
            token_path = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    token = f.read().strip()
                    if token:
                        self.hf_token = token
                        return token
        except:
            pass

        return None

    def test_authentication(self):
        """Verify authentication with Hugging Face Hub using the test script approach"""
        token = self._get_token()

        if not token:
            logging.error("No Hugging Face token found. Please provide a token.")
            return False

        api = HfApi()
        models_to_check = {
            "pyannote/segmentation": "config.yaml",
            "pyannote/speaker-diarization": "config.yaml",
            "pyannote/speaker-diarization-3.1": "config.yaml",
        }

        all_access = True

        for model, test_file in models_to_check.items():
            try:
                info = api.model_info(model, token=token)
                logging.info(f"Can access model info for {model}")

                # Check if the file exists in the list of siblings
                available_files = [s.rfilename for s in info.siblings]
                if test_file not in available_files:
                    # Try to find any config file
                    config_files = [f for f in available_files if f.startswith("config")]
                    if config_files:
                        test_file = config_files[0]
                        logging.info(f"Using alternative file: {test_file}")
                    else:
                        logging.warning(f"File '{test_file}' not found in model '{model}'")
                        all_access = False
                        continue

                # Try downloading the file
                try:
                    hf_hub_download(model, test_file, token=token, local_files_only=False)
                    logging.info(f"Successfully downloaded '{test_file}' from {model}")
                except Exception as e:
                    logging.error(f"Failed to download '{test_file}' from {model}: {e}")
                    all_access = False

            except Exception as e:
                logging.error(f"Cannot access model {model}: {e}")
                all_access = False

        return all_access

    def initialize_pipeline(self):
        """Initialize the diarization pipeline with optimizations"""
        if self.pipeline is not None:
            return True

        # Get the token
        token = self._get_token()
        if not token:
            raise ValueError("No Hugging Face token found. Please provide a valid token.")

        # Get the model to use
        model_id = self.get_model_id()
        logging.info(f"Loading speaker diarization pipeline '{model_id}'...")

        try:
            # Load the pipeline directly
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

            # Move to MPS device
            pipeline = pipeline.to(torch.device("mps"))

            # Apply optimizations similar to working script
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    # Optimize embedding batch size
                    if hasattr(pipeline, "embedding"):
                        # Optimal batch size for M1/M2
                        pipeline.embedding.batch_size = 32

                    # Optimize clustering
                    if hasattr(pipeline, "clustering"):
                        if hasattr(pipeline.clustering, "method"):
                            pipeline.clustering.method = "centroid"

                    logging.info("Applied MPS-specific optimizations")
                except Exception as e:
                    logging.warning(f"Could not apply all optimizations: {e}")

            self.pipeline = pipeline
            logging.info("Speaker diarization pipeline loaded successfully")
            return True

        except Exception as e:
            error_msg = str(e)

            if "401" in error_msg or "unauthorized" in error_msg.lower():
                logging.error("Authentication failed. Please check your token.")
                raise ValueError("Invalid token or authentication failed.")

            elif "403" in error_msg or "cannot access gated repo" in error_msg.lower():
                logging.error(f"You need to accept the model license on Hugging Face Hub.")
                raise ValueError(
                    f"Please visit https://huggingface.co/{model_id.split('@')[0]} "
                    f"to accept the license agreement."
                )

            elif "found in model" in error_msg.lower() or "not found" in error_msg.lower():
                logging.error(f"Required files not found in the model repository.")
                if "Default" in self.model_name and "@" not in model_id:
                    # Try legacy model instead
                    logging.info("Trying legacy model instead...")
                    self.model_name = "Legacy"
                    return self.initialize_pipeline()

            else:
                logging.error(f"Failed to load model: {e}")
                raise

    def diarize(self, audio_path, min_speakers=None, max_speakers=None):
        """Perform speaker diarization on an audio file using the approach from the working script"""
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize the pipeline
        self.initialize_pipeline()

        try:
            logging.info(f"Processing {audio_path}...")

            # Convert min/max speakers to num_speakers parameter
            num_speakers = max_speakers
            if num_speakers is not None:
                logging.info(f"Using fixed number of speakers: {num_speakers}")

            # Use the same process_with_progress approach that works in the script
            diarization = process_with_progress(
                audio_path, self.pipeline, num_speakers=num_speakers
            )

            # Extract segments from the diarization result
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker.replace("SPEAKER_", "Speaker "),
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )

            num_speakers = len(set(s["speaker"] for s in segments))
            logging.info(f"Found {num_speakers} speakers and {len(segments)} segments")

            return segments

        except Exception as e:
            logging.error(f"Diarization failed: {e}")
            raise

    def diarize_and_split(
        self, audio_path, min_speakers=None, max_speakers=None, output_dir=None, export_json=True
    ):
        """Diarize audio and split it into speaker segments

        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            output_dir: Directory to save the split audio segments (optional)
            export_json: Whether to export diarization results to JSON (default: True)

        Returns:
            Tuple of (segments, segment_files) where segments is a list of diarization
            segments and segment_files is a dictionary mapping segment indices to file paths
        """
        # Perform diarization
        segments = self.diarize(audio_path, min_speakers, max_speakers)

        # Create output directory for segments if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            logging.info(f"Created temporary output directory: {output_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Splitting audio into segments in: {output_dir}")

        # Load the audio file
        audio = AudioSegment.from_file(audio_path)

        # Split the audio into segments
        segment_files = {}
        for i, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            speaker = segment["speaker"]

            # Extract segment
            audio_segment = audio[start_ms:end_ms]

            # Create segment file path
            filename = f"{i:04d}_{speaker}_{start_ms}_{end_ms}.wav"
            segment_path = os.path.join(output_dir, filename)

            # Export segment
            audio_segment.export(segment_path, format="wav")

            # Store segment file path
            segment_files[i] = segment_path

            # Add file path to segment data
            segments[i]["file"] = segment_path

        # Export diarization results to JSON if requested
        if export_json:
            json_path = os.path.join(output_dir, "diarization.json")
            with open(json_path, "w") as f:
                json.dump(segments, f, indent=2)
            logging.info(f"Exported diarization data to: {json_path}")

        return segments, segment_files

    def merge_transcripts_with_diarization(self, segments, transcripts):
        """Merge transcription results with diarization segments

        Args:
            segments: List of diarization segments
            transcripts: Dictionary mapping segment indices to transcription text

        Returns:
            Merged transcript with speaker labels
        """
        # Ensure segments are sorted by start time
        sorted_segments = sorted(enumerate(segments), key=lambda x: x[1]["start"])

        # Merge transcripts
        result = []
        current_speaker = None
        current_text = []

        for i, segment in sorted_segments:
            if i not in transcripts:
                continue

            speaker = segment["speaker"]
            text = transcripts[i].strip()

            if not text:
                continue

            if speaker != current_speaker:
                # New speaker
                if current_speaker is not None and current_text:
                    result.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = speaker
                current_text = [text]
            else:
                # Same speaker
                current_text.append(text)

        # Add the last speaker's text
        if current_speaker is not None and current_text:
            result.append(f"{current_speaker}: {' '.join(current_text)}")

        return "\n\n".join(result)

    def export_to_formats(self, segments, transcripts, output_path_base, formats=None):
        """Export diarized transcription to various formats

        Args:
            segments: List of diarization segments
            transcripts: Dictionary mapping segment indices to transcription text
            output_path_base: Base path for output files (without extension)
            formats: List of formats to export (default: ["txt", "srt", "vtt"])

        Returns:
            Dictionary mapping formats to output file paths
        """
        if formats is None:
            formats = ["txt", "srt", "vtt"]

        # Ensure segments are sorted by start time
        sorted_segments = sorted(enumerate(segments), key=lambda x: x[1]["start"])

        # Create mapping of exported files
        exported_files = {}

        # Process each requested format
        for fmt in formats:
            if fmt.lower() == "txt":
                # Plain text format
                txt_path = f"{output_path_base}.txt"
                merged_transcript = self.merge_transcripts_with_diarization(segments, transcripts)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(merged_transcript)
                exported_files["txt"] = txt_path

            elif fmt.lower() == "srt":
                # SRT format
                srt_path = f"{output_path_base}.srt"
                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, (idx, segment) in enumerate(sorted_segments, 1):
                        if idx not in transcripts:
                            continue

                        text = transcripts[idx].strip()
                        if not text:
                            continue

                        start = segment["start"]
                        end = segment["end"]
                        speaker = segment["speaker"]

                        # Format timestamps as HH:MM:SS,mmm
                        start_time = time_to_srt(start)
                        end_time = time_to_srt(end)

                        # Write SRT entry
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{speaker}: {text}\n\n")

                exported_files["srt"] = srt_path

            elif fmt.lower() == "vtt":
                # VTT format (WebVTT)
                vtt_path = f"{output_path_base}.vtt"
                with open(vtt_path, "w", encoding="utf-8") as f:
                    # Write VTT header
                    f.write("WEBVTT\n\n")

                    for i, (idx, segment) in enumerate(sorted_segments, 1):
                        if idx not in transcripts:
                            continue

                        text = transcripts[idx].strip()
                        if not text:
                            continue

                        start = segment["start"]
                        end = segment["end"]
                        speaker = segment["speaker"]

                        # Format timestamps as HH:MM:SS.mmm
                        start_time = time_to_vtt(start)
                        end_time = time_to_vtt(end)

                        # Write VTT entry
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{speaker}: {text}\n\n")

                exported_files["vtt"] = vtt_path

            elif fmt.lower() == "json":
                # JSON format
                json_path = f"{output_path_base}.json"
                output_data = []

                for idx, segment in sorted_segments:
                    if idx not in transcripts:
                        continue

                    text = transcripts[idx].strip()
                    if not text:
                        continue

                    output_data.append(
                        {
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": text,
                        }
                    )

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)

                exported_files["json"] = json_path

        return exported_files


def time_to_srt(seconds):
    """Convert time in seconds to SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def time_to_vtt(seconds):
    """Convert time in seconds to VTT format (HH:MM:SS.mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def verify_authentication(token=None):
    """Test connection to Hugging Face - directly using the proven test script approach"""
    if not token:
        token = os.environ.get("HF_TOKEN")

    if not token:
        try:
            token_path = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    token = f.read().strip()
        except:
            pass

    if not token:
        print(
            "❌ No token found. Please provide a token via argument, environment variable, or ~/.huggingface/token file."
        )
        return False

    try:
        # We DON'T call login() here directly - not needed for testing
        print(f"Token length: {len(token)} characters")
    except Exception as e:
        print(f"❌ Token error: {e}")
        return False

    api = HfApi()
    models_to_check = {
        "pyannote/segmentation": "config.yaml",
        "pyannote/speaker-diarization": "config.yaml",
        "pyannote/speaker-diarization-3.1": "config.yaml",
    }

    all_access = True

    for model, test_file in models_to_check.items():
        try:
            info = api.model_info(model, token=token)
            print(f"✅ Can access model info for {model}")

            # Check if the file exists in the list of siblings
            available_files = [s.rfilename for s in info.siblings]
            if test_file not in available_files:
                print(
                    f"  ⚠️ File '{test_file}' not found in model '{model}' -- available files: {available_files[:5]}..."
                )

                # Try to find a config file
                config_files = [f for f in available_files if f.startswith("config")]
                if config_files:
                    test_file = config_files[0]
                    print(f"  ℹ️ Will try alternative file: {test_file}")
                else:
                    all_access = False
                    continue

            # Try downloading the file
            try:
                hf_hub_download(model, test_file, token=token, local_files_only=False)
                print(f"  ✅ Successfully downloaded '{test_file}' from {model}")
            except Exception as e:
                print(f"  ❌ Failed to download '{test_file}' from {model}: {e}")
                all_access = False

        except Exception as e:
            print(f"❌ Cannot access model {model}: {e}")
            all_access = False
            if "403" in str(e) or "401" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"  ⚠️ This model is gated. Visit and request access:")
                print(f"  🔗 https://huggingface.co/{model}")

    if all_access:
        print("\n✅ All models can be accessed! Your authentication is working correctly.")
        return True
    else:
        print("\n⚠️ Some models could not be accessed properly.")
        print("Please ensure you've accepted any required license agreements or requested access:")
        for model in models_to_check:
            print(f"- https://huggingface.co/{model}")
        return False
