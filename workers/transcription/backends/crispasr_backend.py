# workers/transcription/backends/crispasr_backend.py
"""CrispASR backend — unified multi-model ASR via the crispasr binary.

Supports all CrispASR backends (parakeet, canary, cohere, granite,
qwen3, voxtral, voxtral4b, fastconformer-ctc, wav2vec2) through a
single interface. The backend auto-detects from the GGUF file, or
can be forced with the `crispasr_backend` kwarg.

Requires the `crispasr` binary on PATH or at CRISPASR_EXECUTABLE.
Build from https://github.com/CrispStrobe/CrispASR
"""
import logging
import os
import re
import subprocess
import threading
from .base import TranscriptionBackend


_GITHUB_RELEASE_URL = "https://github.com/CrispStrobe/CrispASR/releases/latest/download"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "susurrus", "crispasr")


def _download_crispasr():
    """Download the latest CrispASR release for this platform."""
    import platform
    import zipfile
    import tarfile
    import urllib.request

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        asset = "crispasr-linux-x86_64.tar.gz"
    elif system == "darwin":
        asset = "crispasr-macos.tar.gz"
    elif system == "windows":
        asset = "crispasr-windows-x86_64.zip"
    else:
        logging.warning(f"No pre-built CrispASR binary for {system}/{machine}")
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)
    exe_name = "crispasr.exe" if system == "windows" else "crispasr"
    cached_exe = os.path.join(_CACHE_DIR, exe_name)

    if os.path.isfile(cached_exe) and os.access(cached_exe, os.X_OK):
        return cached_exe

    url = f"{_GITHUB_RELEASE_URL}/{asset}"
    archive_path = os.path.join(_CACHE_DIR, asset)
    logging.info(f"Downloading CrispASR from {url} ...")

    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        logging.warning(f"Failed to download CrispASR: {e}")
        return None

    try:
        if asset.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(_CACHE_DIR)
        elif asset.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(_CACHE_DIR)
    except Exception as e:
        logging.warning(f"Failed to extract CrispASR: {e}")
        return None
    finally:
        if os.path.isfile(archive_path):
            os.remove(archive_path)

    # The archive contains a subdirectory — find the binary
    for root, _dirs, files in os.walk(_CACHE_DIR):
        for f in files:
            if f == exe_name:
                path = os.path.join(root, f)
                if system != "windows":
                    os.chmod(path, 0o755)
                return path

    logging.warning("CrispASR binary not found in downloaded archive")
    return None


def _find_crispasr():
    """Locate the crispasr binary — search PATH, common locations, then auto-download."""
    # Explicit env var
    env = os.environ.get("CRISPASR_EXECUTABLE")
    if env and os.path.isfile(env):
        return env

    # Common install locations
    candidates = [
        "crispasr",  # on PATH
        os.path.expanduser("~/.local/bin/crispasr"),
        "/usr/local/bin/crispasr",
    ]
    # Also check whisper.cpp / CrispASR build dirs
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "whisper.cpp"),
        os.path.expanduser("~/whisper.cpp"),
        os.path.expanduser("~/CrispASR"),
    ]:
        candidates.append(os.path.join(base, "build", "bin", "crispasr"))

    # Check cached download
    exe_name = "crispasr.exe" if os.name == "nt" else "crispasr"
    candidates.append(os.path.join(_CACHE_DIR, exe_name))
    # Also check subdirectory from archive extraction
    for sub in ("crispasr-linux-x86_64", "crispasr-macos", "crispasr-windows-x86_64"):
        candidates.append(os.path.join(_CACHE_DIR, sub, exe_name))

    import shutil
    for c in candidates:
        if c == "crispasr":
            found = shutil.which(c)
            if found:
                return found
        elif os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # Not found anywhere — try to download
    logging.info("CrispASR not found locally, attempting to download latest release...")
    return _download_crispasr()


class CrispasrBackend(TranscriptionBackend):
    """CrispASR backend — calls the crispasr binary for any supported model.

    model_id should be a path to a GGUF model file. The backend is
    auto-detected from GGUF metadata, or forced via crispasr_backend kwarg.

    Kwargs:
        crispasr_backend: str — force a specific backend (e.g. "parakeet")
        word_timestamps: bool — request word-level timestamps
        vad: bool — enable Silero VAD for long audio
        split_on_punct: bool — split subtitles at sentence boundaries
        temperature: float — sampling temperature (0 = greedy)
        best_of: int — best-of-N candidates with temperature > 0
    """

    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.crispasr_backend = kwargs.get("crispasr_backend", None)
        self.vad = kwargs.get("vad", False)
        self.split_on_punct = kwargs.get("split_on_punct", False)
        self.temperature = kwargs.get("temperature", 0.0)
        self.best_of = kwargs.get("best_of", 1)
        self.temp_files = []

    def preprocess_audio(self, audio_path):
        """Convert to WAV if needed (crispasr handles WAV/MP3/FLAC/OGG)."""
        # CrispASR can decode most formats via miniaudio, but WAV is safest
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".ogg"):
            return audio_path
        from utils.audio_utils import convert_audio_to_wav
        wav_path = convert_audio_to_wav(audio_path)
        if wav_path != audio_path:
            self.temp_files.append(wav_path)
        return wav_path

    def transcribe(self, audio_path):
        """Transcribe using the crispasr binary."""
        logging.info("=== Starting CrispASR pipeline ===")

        exe = _find_crispasr()
        if not exe:
            raise FileNotFoundError(
                "crispasr binary not found. Set CRISPASR_EXECUTABLE or "
                "install from https://github.com/CrispStrobe/CrispASR"
            )
        logging.info(f"Using crispasr: {exe}")

        if not os.path.isfile(self.model_id):
            raise FileNotFoundError(f"Model not found: {self.model_id}")

        cmd = [
            exe,
            "-m", self.model_id,
            "-f", audio_path,
            "-t", str(min(os.cpu_count() or 4, 8)),
            "-np",  # no progress prints on stderr
        ]

        if self.crispasr_backend:
            cmd.extend(["--backend", self.crispasr_backend])

        if self.language:
            cmd.extend(["-l", self.language])

        if self.vad:
            cmd.append("--vad")

        if self.split_on_punct:
            cmd.append("--split-on-punct")

        if self.word_timestamps:
            cmd.extend(["-ml", "1"])  # max_len=1 → one word per segment

        if self.temperature > 0:
            cmd.extend(["-tp", str(self.temperature)])
            if self.best_of > 1:
                cmd.extend(["--best-of", str(self.best_of)])

        logging.info(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        output_lines = []

        def collect_stderr():
            for line in iter(process.stderr.readline, ""):
                line = line.strip()
                if line:
                    logging.info(f"crispasr: {line}")

        stderr_thread = threading.Thread(target=collect_stderr, daemon=True)
        stderr_thread.start()

        # Read stdout — crispasr outputs [HH:MM:SS.mmm --> HH:MM:SS.mmm] text
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r"\[(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]\s*(.*)",
                line,
            )
            if m:
                start = self._parse_ts(m.group(1))
                end = self._parse_ts(m.group(2))
                text = m.group(3).strip()
                if text:
                    output_lines.append((start, end, text))
            else:
                # Plain text line (no timestamps)
                if line:
                    output_lines.append((0.0, 0.0, line))

        rc = process.wait()
        stderr_thread.join(timeout=2)

        if rc != 0:
            logging.error(f"crispasr failed with code {rc}")
            raise RuntimeError(f"crispasr exited with code {rc}")

        for result in output_lines:
            yield result

    @staticmethod
    def _parse_ts(ts_str):
        """Parse HH:MM:SS.mmm to seconds."""
        parts = ts_str.split(":")
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

    def cleanup(self):
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {f}: {e}")
