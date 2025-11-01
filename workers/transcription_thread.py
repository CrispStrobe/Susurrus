#!/usr/bin/env python3
"""Transcription worker thread"""
import json
import logging
import os
import re
import subprocess
import sys
import threading

from PyQt6.QtCore import QThread, pyqtSignal

# Configure logging
logger = logging.getLogger(__name__)


class TranscriptionThread(QThread):
    """Thread for running transcription workers"""

    progress_signal = pyqtSignal(str, str)  # metrics, transcription
    error_signal = pyqtSignal(str)
    transcription_replace_signal = pyqtSignal(str)
    diarization_signal = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._is_running = True
        self.process = None

    def format_time(self, time_str):
        """Format time string"""
        if not time_str:
            return ""
        try:
            time_str = time_str.replace(",", ".")
            return f"{float(time_str):.3f}"
        except ValueError:
            return None

    def run(self):
        """Main thread execution"""
        try:
            if self.args.get("diarization_enabled", False):
                self._run_diarization()
            else:
                self._run_standard_transcription()
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Use logger instead of logging
            self.error_signal.emit(error_msg)

    def _run_diarization(self):
        """Run transcription with speaker diarization"""
        try:
            # Check if diarize_audio module exists
            try:
                import importlib.util

                spec = importlib.util.find_spec("backends.diarization.manager")
                if spec is None:
                    raise ImportError("Diarization module not found")
            except ImportError as e:
                error_msg = f"Failed to import diarization module: {str(e)}"
                logger.error(error_msg)
                self.error_signal.emit(error_msg)
                return

            # Extract arguments
            audio_input = self.args["audio_input"]
            hf_token = self.args["hf_token"]
            model_id = self.args["model_id"]
            language = self.args["language"]
            backend = self.args["backend"]
            device_arg = self.args["device_arg"]
            min_speakers = self.args.get("min_speakers")
            max_speakers = self.args.get("max_speakers")
            output_format = self.args.get("output_format", "txt")
            diarization_model = self.args.get("diarization_model", "Default")

            python_executable = sys.executable

            # Get the correct path to diarize_worker.py
            # It should be in the same directory as this file (workers/)
            worker_dir = os.path.dirname(os.path.abspath(__file__))
            diarize_worker_path = os.path.join(worker_dir, "diarize_worker.py")

            if not os.path.exists(diarize_worker_path):
                # Try parent directory (for backward compatibility)
                diarize_worker_path = os.path.join(os.path.dirname(worker_dir), "diarize_worker.py")
                if not os.path.exists(diarize_worker_path):
                    self.error_signal.emit("diarize_worker.py not found")
                    return

            logger.info(f"Using diarize_worker.py at: {diarize_worker_path}")

            # Build the command
            cmd = [
                python_executable,
                "-u",
                diarize_worker_path,
                "--audio-input",
                audio_input,
                "--hf-token",
                hf_token,
                "--transcribe",
                "--model-id",
                model_id,
                "--backend",
                backend,
                "--device",
                device_arg,
                "--output-formats",
                output_format,
                "--diarization-model",
                diarization_model,
            ]

            if min_speakers:
                cmd.extend(["--min-speakers", str(min_speakers)])

            if max_speakers:
                cmd.extend(["--max-speakers", str(max_speakers)])

            if language:
                cmd.extend(["--language", language])

            self.progress_signal.emit("Starting speaker diarization...", "")
            logger.info(f"Running command: {' '.join(cmd)}")

            # Run the diarization worker
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, text=True
            )

            # Track output files
            output_files = {}
            diarization_json = None

            # Read from stderr in separate thread
            def read_stderr():
                for line in iter(self.process.stderr.readline, ""):
                    if not self._is_running:
                        break
                    self.progress_signal.emit(line.strip(), "")

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Process stdout
            for line in iter(self.process.stdout.readline, ""):
                if not self._is_running:
                    break

                line = line.strip()

                if line.startswith("OUTPUT FILE ("):
                    match = re.match(r"OUTPUT FILE \((.+)\): (.+)", line)
                    if match:
                        format_type, file_path = match.groups()
                        output_files[format_type.lower()] = file_path
                        self.progress_signal.emit(
                            f"Generated {format_type} transcript: {file_path}", ""
                        )

                elif line.startswith("DIARIZATION JSON:"):
                    diarization_json = line.split(":", 1)[1].strip()
                    self.progress_signal.emit(f"Diarization data saved to: {diarization_json}", "")

                else:
                    self.progress_signal.emit(line, "")

            # Wait for completion
            self.process.wait()
            stderr_thread.join(timeout=1)

            if self.process.returncode != 0:
                self.error_signal.emit("Speaker diarization process failed")
                return

            # Load and display transcript
            if output_format.lower() in output_files:
                file_path = output_files[output_format.lower()]
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        transcript = f.read()
                        self.transcription_replace_signal.emit(transcript)
                except Exception as e:
                    self.error_signal.emit(f"Failed to read transcript: {str(e)}")

            # Load diarization info
            if diarization_json:
                try:
                    with open(diarization_json, "r", encoding="utf-8") as f:
                        diarization_data = json.load(f)
                        speaker_count = len(set(seg["speaker"] for seg in diarization_data))
                        segment_count = len(diarization_data)
                        self.diarization_signal.emit(
                            f"Diarization successful. Found {speaker_count} "
                            f"speakers in {segment_count} segments."
                        )
                except Exception as e:
                    self.error_signal.emit(f"Failed to read diarization data: {str(e)}")

        except Exception as e:
            error_msg = f"Diarization error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)

    def _run_standard_transcription(self):
        """Run standard transcription without diarization"""
        try:
            # Extract arguments
            model_id = self.args["model_id"]
            word_timestamps = self.args.get("word_timestamps", False)
            language = self.args["language"]
            backend = self.args["backend"]
            device_arg = self.args["device_arg"]
            pipeline_type = self.args.get("pipeline_type", "default")
            max_chunk_length = float(self.args.get("max_chunk_length", 0))
            output_format = self.args.get("output_format", "txt")
            quantization = self.args.get("quantization")
            start_time = self.format_time(self.args.get("start_time", ""))
            end_time = self.format_time(self.args.get("end_time", ""))

            if start_time is None or end_time is None:
                start_time = ""
                end_time = ""

            python_executable = sys.executable

            # Get correct path to transcribe_worker.py
            worker_dir = os.path.dirname(os.path.abspath(__file__))
            transcribe_worker_path = os.path.join(worker_dir, "transcribe_worker.py")

            if not os.path.exists(transcribe_worker_path):
                transcribe_worker_path = os.path.join(
                    os.path.dirname(worker_dir), "transcribe_worker.py"
                )
                if not os.path.exists(transcribe_worker_path):
                    self.error_signal.emit("transcribe_worker.py not found")
                    return

            logger.info(f"Using transcribe_worker.py at: {transcribe_worker_path}")

            # Build command
            cmd = [
                python_executable,
                "-u",
                transcribe_worker_path,
                "--model-id",
                model_id,
                "--backend",
                backend,
                "--device",
                device_arg,
            ]

            if self.args.get("preprocessor_path"):
                cmd.extend(["--preprocessor-path", self.args["preprocessor_path"]])

            if self.args.get("original_model_id"):
                cmd.extend(["--original-model-id", self.args["original_model_id"]])

            if self.args.get("audio_input"):
                cmd.extend(["--audio-input", self.args["audio_input"]])
            if self.args.get("audio_url"):
                cmd.extend(["--audio-url", self.args["audio_url"]])

            if self.args.get("proxy_url"):
                cmd.extend(["--proxy-url", self.args["proxy_url"]])
            if self.args.get("proxy_username"):
                cmd.extend(["--proxy-username", self.args["proxy_username"]])
            if self.args.get("proxy_password"):
                cmd.extend(["--proxy-password", self.args["proxy_password"]])

            if word_timestamps:
                cmd.append("--word-timestamps")
            if language:
                cmd.extend(["--language", language])
            if pipeline_type != "default":
                cmd.extend(["--pipeline-type", pipeline_type])
            if max_chunk_length > 0:
                cmd.extend(["--max-chunk-length", str(max_chunk_length)])
            if backend == "whisper.cpp" and output_format:
                cmd.extend(["--output-format", output_format])
            if backend == "ctranslate2" and quantization:
                cmd.extend(["--quantization", quantization])
            if start_time:
                cmd.extend(["--start-time", start_time])
            if end_time:
                cmd.extend(["--end-time", end_time])

            if self.args.get("mistral_api_key"):
                cmd.extend(["--mistral-api-key", self.args["mistral_api_key"]])

            logger.info(f"Running command: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )

            # Read stderr in separate thread
            def read_stderr():
                for line in self.process.stderr:
                    if not self._is_running:
                        break
                    line = line.decode("utf-8", errors="replace").rstrip()
                    if line:
                        self.progress_signal.emit(line, "")

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Process stdout
            timecode_pattern = re.compile(r"^\[([^\]]+?)\]\s*(.*)")
            in_transcription = False
            output_file = None
            is_whisper_jax = self.args["backend"] == "whisper-jax"
            whisper_jax_output = ""

            for line in self.process.stdout:
                if not self._is_running:
                    break
                line = line.decode("utf-8", errors="replace").rstrip()

                if line.startswith("OUTPUT FILE: "):
                    output_file = line[len("OUTPUT FILE: ") :].strip()
                    continue

                if line.startswith("\rProgress:"):
                    self.progress_signal.emit(line, "")
                    continue

                if is_whisper_jax:
                    if any(
                        x in line
                        for x in [
                            "Transcription time:",
                            "Audio file size:",
                            "Total transcription time:",
                        ]
                    ):
                        self.progress_signal.emit(line, "")
                    else:
                        whisper_jax_output += line + "\n"
                        self.progress_signal.emit("", line)
                elif any(
                    x in line
                    for x in [
                        "Starting transcription",
                        "Detected language",
                        "Total transcription time",
                    ]
                ):
                    self.progress_signal.emit(line, "")
                    in_transcription = False
                elif timecode_pattern.match(line):
                    match = timecode_pattern.match(line)
                    text = match.group(2)
                    self.progress_signal.emit("", text)
                    in_transcription = True
                elif in_transcription and line.strip() != "":
                    self.progress_signal.emit("", line)
                elif line.strip() != "":
                    self.progress_signal.emit(line, "")
                    in_transcription = False

            # Wait for completion
            self.process.stdout.close()
            self.process.wait()
            stderr_thread.join(timeout=1)

            if self.process.returncode != 0:
                self.error_signal.emit("Transcription process failed.")
            else:
                if is_whisper_jax:
                    self.transcription_replace_signal.emit(whisper_jax_output)
                elif backend == "whisper.cpp" and output_format in ("srt", "vtt"):
                    if output_file and os.path.exists(output_file):
                        with open(output_file, "r", encoding="utf-8") as f:
                            output_content = f.read()
                            self.transcription_replace_signal.emit(output_content)
                    else:
                        self.error_signal.emit(f"Output file not found: {output_file}")

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)

    def stop(self):
        """Stop the transcription process"""
        self._is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
