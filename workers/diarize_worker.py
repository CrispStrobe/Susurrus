#!/usr/bin/env python3
# diarize_worker.py - Worker script for audio diarization and transcription

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time

# Add parent directory to path so we can import from our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modularized backend
from .manager import DiarizationManager, verify_authentication

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

# Create a custom progress indicator for transcription
class TranscriptionProgress:
    """Simple progress indicator for transcription process"""

    def __init__(self, total_segments):
        self.total = total_segments
        self.completed = 0
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._progress_indicator)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Update progress counter"""
        self.completed += 1

    def _progress_indicator(self):
        """Progress indicator thread"""
        while self.running and self.completed < self.total:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)

            if self.total > 0:
                percentage = (self.completed / self.total) * 100
                remaining = elapsed * (self.total - self.completed) / max(1, self.completed)
                rem_mins, rem_secs = divmod(int(remaining), 60)

                bar_len = 20
                filled_len = int(bar_len * percentage / 100)
                bar = "■" * filled_len + "□" * (bar_len - filled_len)

                print(
                    f"\r[{mins:02d}:{secs:02d}] Transcribing: [{bar}] {percentage:.1f}% ({self.completed}/{self.total}) Est. remaining: {rem_mins}m {rem_secs}s",
                    end="",
                    flush=True,
                )
            else:
                spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][int(elapsed) % 10]
                print(f"\r[{mins:02d}:{secs:02d}] {spinner} Transcribing...", end="", flush=True)

            time.sleep(0.5)

    def finish(self):
        """Complete the progress tracking"""
        self.running = False

        # Clear the progress line
        print("\r" + " " * 100 + "\r", end="", flush=True)

        # Print completion stats
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)

        if self.total > 0:
            avg_time = elapsed / self.total
            print(
                f"✅ Transcription completed in {mins:02d}:{secs:02d} ({avg_time:.2f}s per segment)"
            )
        else:
            print(f"✅ Transcription completed in {mins:02d}:{secs:02d}")


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization Worker for Susurrus")

    # Audio input parameters
    parser.add_argument("--audio-input", help="Path to the audio input file")
    parser.add_argument("--hf-token", help="Hugging Face API token")

    # Diarization parameters
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for diarized segments"
    )
    parser.add_argument(
        "--diarization-model",
        type=str,
        default="Default",
        help="Diarization model to use (Default, Legacy)",
    )

    # Transcription parameters
    parser.add_argument("--transcribe", action="store_true", help="Transcribe diarized segments")
    parser.add_argument(
        "--model-id", type=str, default="base", help="Whisper model ID to use for transcription"
    )
    parser.add_argument("--language", default=None, help="Language code for transcription")
    parser.add_argument(
        "--backend", type=str, default="transformers", help="Backend to use for transcription"
    )

    # Output format parameters
    parser.add_argument(
        "--output-formats", type=str, default="txt,srt,vtt", help="Output formats (comma separated)"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Base path for output files (without extension)",
    )

    # Authentication verification
    parser.add_argument(
        "--verify-auth",
        action="store_true",
        help="Verify Hugging Face authentication before processing",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.audio_input and not args.verify_auth:
        parser.error("--audio-input is required unless --verify-auth is specified")

    if args.audio_input and not os.path.exists(args.audio_input):
        parser.error(f"Audio file not found: {args.audio_input}")

    # Import necessary modules
    try:
        logging.info("Importing necessary modules...")
    except ImportError as e:
        logging.error(f"Failed to import diarize_audio module: {e}")
        logging.error("Please create diarize_audio.py and diarize_worker.py files")
        sys.exit(1)

    # Verify authentication if requested
    if args.verify_auth:
        logging.info("Verifying Hugging Face authentication...")
        if verify_authentication(args.hf_token):
            logging.info("Authentication successful! Your token has the necessary access.")
            sys.exit(0)
        else:
            logging.error("Authentication failed. Please check your token and model access.")
            sys.exit(1)

    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Determine output base path
    output_base = args.output_base
    if not output_base:
        output_base = os.path.join(
            output_dir, os.path.splitext(os.path.basename(args.audio_input))[0]
        )

    # Parse output formats
    output_formats = [fmt.strip().lower() for fmt in args.output_formats.split(",")]

    # Check if diarization model is valid
    available_models = DiarizationManager.list_available_models()
    if args.diarization_model not in available_models:
        valid_models = ", ".join(available_models)
        logging.warning(
            f"Invalid diarization model: {args.diarization_model}. Valid options are: {valid_models}"
        )
        logging.warning(f"Falling back to Default model.")
        diarization_model = "Default"
    else:
        diarization_model = args.diarization_model

    # Initialize diarization manager with the selected model
    try:
        logging.info("Starting speaker diarization...")
        # Create the diarization manager
        diarization_manager = DiarizationManager(
            hf_token=args.hf_token, device=args.device, model_name=diarization_model
        )

        # Start diarization
        logging.info(
            f"Starting diarization for {args.audio_input} using model: {diarization_model}"
        )
        start_time = time.time()

        try:
            segments, segment_files = diarization_manager.diarize_and_split(
                args.audio_input,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                output_dir=output_dir,
            )

            diarization_time = time.time() - start_time
            logging.info(f"Diarization completed in {diarization_time:.2f} seconds")
            num_speakers = len(set(s["speaker"] for s in segments))
            logging.info(f"Found {num_speakers} speakers in {len(segments)} segments")

            # Export diarization results
            diarization_json = os.path.join(output_dir, "diarization.json")
            with open(diarization_json, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2)
            logging.info(f"Exported diarization data to: {diarization_json}")

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Diarization error: {error_msg}")

            # Handle specific error types
            if "Invalid user token" in error_msg or "401" in error_msg:
                logging.error("Authentication failed with Hugging Face. Please:")
                logging.error("1. Get a token from https://huggingface.co/settings/tokens")
                logging.error("2. Make sure you've accepted the model licenses:")
                model_id = diarization_manager.get_model_id()
                logging.error(f"   - https://huggingface.co/{model_id.split('@')[0]}")
                logging.error(f"   - https://huggingface.co/pyannote/segmentation")
                logging.error(
                    "3. Try again with the --verify-auth flag to test your authentication"
                )

                # Suggest using the verify-auth command
                logging.error("\nTry running with the verify-auth flag:")
                logging.error(f"    python diarize_worker.py --verify-auth --hf-token YOUR_TOKEN")
                sys.exit(1)

            elif (
                "license" in error_msg.lower()
                or "403" in error_msg
                or "access" in error_msg.lower()
            ):
                model_id = diarization_manager.get_model_id()
                logging.error(f"You need to accept the model license on Hugging Face Hub.")
                logging.error(f"Please visit:")
                logging.error(f"1. https://huggingface.co/{model_id.split('@')[0]}")
                logging.error(f"2. https://huggingface.co/pyannote/segmentation")
                logging.error(
                    "Accept the user conditions on both pages while logged in to your account."
                )
                sys.exit(1)

            else:
                # Re-raise other exceptions
                raise

        # Transcribe segments if requested
        if args.transcribe:
            logging.info("Starting transcription of diarized segments")
            transcription_start = time.time()

            # Initialize progress tracker
            progress = TranscriptionProgress(len(segments))

            # Run transcription on each segment using transcribe_worker.py
            transcripts = {}
            python_executable = sys.executable

            for i, segment in enumerate(segments):
                segment_file = segment["file"]
                speaker = segment["speaker"]
                start = segment["start"]
                end = segment["end"]

                # Prepare a temporary file to capture transcription output
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                temp_output.close()

                # Set language
                if args.language:
                    language_arg = args.language
                else:
                    # set a default based on mostly expected content
                    language_arg = "de"  # German as default, change if needed

                # Build the command to run transcribe_worker.py
                cmd = [
                    python_executable,
                    "transcribe_worker.py",
                    "--audio-input",
                    segment_file,
                    "--model-id",
                    args.model_id,
                    "--backend",
                    args.backend,
                    "--device",
                    args.device or "auto",
                    "--language",
                    language_arg,  # Always specify language to skip detection
                ]

                # Add optional arguments
                if args.language:
                    cmd.extend(["--language", args.language])

                try:
                    # Run transcribe_worker.py on this segment
                    process = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                    )

                    # Extract the transcription text from the output
                    lines = process.stdout.strip().split("\n")
                    text_lines = []

                    # Parse the output - improved filtering to remove language detection messages
                    for line in lines:
                        # Skip language detection lines and similar non-transcription content
                        if any(
                            skip_text in line
                            for skip_text in [
                                "Detecting language",
                                "Use the `language`",
                                "Detected language:",
                                "OUTPUT FILE:",
                                "Transcription time:",
                            ]
                        ):
                            continue

                        # Check if line contains timestamps
                        if "[" in line and "]" in line and "-->" in line:
                            # Extract text after timestamp format: [00:00:00.000 --> 00:00:05.000] Text here
                            text_part = line.split("]", 1)[1].strip() if "]" in line else line
                            text_lines.append(text_part)
                        elif line.strip():  # Only add non-empty lines
                            # Regular text line without timestamps
                            text_lines.append(line.strip())

                    # Join all the text lines
                    text = " ".join(text_lines).strip()

                    # Remove any duplicated text that might occur due to repeated transcription
                    if text and len(text) > 5:
                        # This handles cases where the same text appears twice
                        half_length = len(text) // 2
                        first_half = text[:half_length].strip()
                        second_half = text[half_length:].strip()

                        if first_half and first_half == second_half:
                            text = first_half  # Use just one copy if duplicated
                    transcripts[i] = text

                    # Log transcript for first few segments only
                    if i < 5:
                        logging.info(f"Transcript {i+1}: {text[:100]}...")

                    # Update progress
                    progress.update()

                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to transcribe segment {i}: {e}")
                    logging.error(f"Error output: {e.stderr}")
                    transcripts[i] = ""
                    progress.update()
                except Exception as e:
                    logging.error(f"Failed to transcribe segment {i}: {e}")
                    transcripts[i] = ""
                    progress.update()

            # Finish progress indicator
            progress.finish()

            transcription_time = time.time() - transcription_start
            logging.info(f"Transcription completed in {transcription_time:.2f} seconds")

            # Export transcripts
            transcripts_json = os.path.join(output_dir, "transcripts.json")
            with open(transcripts_json, "w", encoding="utf-8") as f:
                json.dump(transcripts, f, indent=2)
            logging.info(f"Exported transcripts to: {transcripts_json}")

            # Export merged transcript in requested formats
            try:
                logging.info(f"Exporting transcripts in formats: {', '.join(output_formats)}")
                exported_files = diarization_manager.export_to_formats(
                    segments, transcripts, output_base, formats=output_formats
                )

                for fmt, file_path in exported_files.items():
                    logging.info(f"Exported {fmt.upper()} transcript to: {file_path}")

                    # Print output file path for parent process
                    print(f"OUTPUT FILE ({fmt.upper()}): {file_path}")
            except Exception as e:
                logging.error(f"Failed to export transcripts: {e}")

        # Print overall stats
        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")

        # Print diarization results JSON path for parent process
        print(f"DIARIZATION JSON: {diarization_json}")

    except Exception as e:
        logging.error(f"Speaker diarization process failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
