"""Shared utilities for locating, downloading, and probing the CrispASR binary."""

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess

_GITHUB_RELEASE_URL = "https://github.com/CrispStrobe/CrispASR/releases/latest/download"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "susurrus", "crispasr")

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Binary discovery & download
# --------------------------------------------------------------------------


def download_crispasr():
    """Download the latest CrispASR release for this platform."""
    import tarfile
    import urllib.request
    import zipfile

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        asset = "crispasr-linux-x86_64.tar.gz"
    elif system == "darwin":
        asset = "crispasr-macos.tar.gz"
    elif system == "windows":
        asset = "crispasr-windows-x86_64.zip"
    else:
        logger.warning("No pre-built CrispASR binary for %s/%s", system, machine)
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)
    exe_name = "crispasr.exe" if system == "windows" else "crispasr"
    cached_exe = os.path.join(_CACHE_DIR, exe_name)

    if os.path.isfile(cached_exe) and os.access(cached_exe, os.X_OK):
        return cached_exe

    url = f"{_GITHUB_RELEASE_URL}/{asset}"
    archive_path = os.path.join(_CACHE_DIR, asset)
    logger.info("Downloading CrispASR from %s ...", url)

    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        logger.warning("Failed to download CrispASR: %s", e)
        return None

    try:
        if asset.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(_CACHE_DIR, filter="data")
        elif asset.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.namelist():
                    if os.path.isabs(member) or ".." in member.split("/"):
                        raise ValueError(f"Unsafe zip member: {member}")
                zf.extractall(_CACHE_DIR)  # nosec B202
    except Exception as e:
        logger.warning("Failed to extract CrispASR: %s", e)
        return None
    finally:
        if os.path.isfile(archive_path):
            os.remove(archive_path)

    for root, _dirs, files in os.walk(_CACHE_DIR):
        for f in files:
            if f == exe_name:
                path = os.path.join(root, f)
                if system != "windows":
                    os.chmod(path, 0o755)
                return path

    logger.warning("CrispASR binary not found in downloaded archive")
    return None


def find_crispasr():
    """Locate the crispasr binary — search PATH, common locations, then auto-download."""
    env = os.environ.get("CRISPASR_EXECUTABLE")
    if env and os.path.isfile(env):
        return env

    candidates = [
        "crispasr",
        os.path.expanduser("~/.local/bin/crispasr"),
        "/usr/local/bin/crispasr",
    ]
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "whisper.cpp"),
        os.path.expanduser("~/whisper.cpp"),
        os.path.expanduser("~/CrispASR"),
    ]:
        candidates.append(os.path.join(base, "build", "bin", "crispasr"))

    exe_name = "crispasr.exe" if os.name == "nt" else "crispasr"
    candidates.append(os.path.join(_CACHE_DIR, exe_name))
    for sub in ("crispasr-linux-x86_64", "crispasr-macos", "crispasr-windows-x86_64"):
        candidates.append(os.path.join(_CACHE_DIR, sub, exe_name))

    for c in candidates:
        if c == "crispasr":
            found = shutil.which(c)
            if found:
                return found
        elif os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    logger.info("CrispASR not found locally, attempting to download latest release...")
    return download_crispasr()


# --------------------------------------------------------------------------
# Backend probing (like CrisperWeaver's availableBackends() check)
# --------------------------------------------------------------------------

_cached_backends = None
_cached_backend_caps = {}
_cached_version = None


def probe_backends(exe=None):
    """Probe which backends the CrispASR binary was built with.

    Returns a list of backend name strings, or an empty list on failure.
    Results are cached for the lifetime of the process.
    """
    global _cached_backends
    if _cached_backends is not None:
        return _cached_backends

    if exe is None:
        exe = find_crispasr()
    if not exe:
        _cached_backends = []
        return _cached_backends

    try:
        import json

        result = subprocess.run(
            [exe, "--list-backends-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            if isinstance(data, list):
                # Rich format: [{"name": "whisper", "caps": [...], ...}, ...]
                if data and isinstance(data[0], dict):
                    _cached_backends = [entry["name"] for entry in data if "name" in entry]
                    # Also cache the full capability info
                    _cached_backend_caps = {
                        entry["name"]: entry.get("caps", [])
                        for entry in data if "name" in entry
                    }
                else:
                    _cached_backends = data
            elif isinstance(data, dict) and "backends" in data:
                _cached_backends = data["backends"]
            else:
                _cached_backends = []
        else:
            # Fallback: try --list-backends (plain text)
            result = subprocess.run(
                [exe, "--list-backends"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                _cached_backends = [
                    line.strip() for line in result.stdout.splitlines()
                    if line.strip() and not line.startswith("Available")
                ]
            else:
                _cached_backends = []
    except Exception as e:
        logger.warning("Failed to probe CrispASR backends: %s", e)
        _cached_backends = []

    return _cached_backends


def probe_version(exe=None):
    """Get CrispASR version string. Cached."""
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    if exe is None:
        exe = find_crispasr()
    if not exe:
        _cached_version = "unknown"
        return _cached_version

    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            _cached_version = result.stdout.strip().split("\n")[0]
        else:
            _cached_version = "unknown"
    except Exception:
        _cached_version = "unknown"

    return _cached_version


def is_backend_available(backend_name, exe=None):
    """Check if a specific backend is available in the binary."""
    backends = probe_backends(exe)
    return backend_name in backends


def backend_capabilities(backend_name, exe=None):
    """Return the capability list for a backend, or empty list.

    Capabilities are strings like 'timestamps-native', 'word-timestamps',
    'auto-download', 'temperature', 'translate', etc.
    """
    probe_backends(exe)  # Ensure cache is populated
    return list(_cached_backend_caps.get(backend_name, []))


# --------------------------------------------------------------------------
# SHA-256 verification for downloaded files
# --------------------------------------------------------------------------


def verify_sha256(filepath, expected_hash):
    """Verify SHA-256 hash of a file.

    Returns True if the hash matches, False otherwise.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_hash:
        logger.warning(
            "SHA-256 mismatch for %s: expected %s, got %s",
            filepath, expected_hash, actual,
        )
        return False
    return True


def download_model(url, filename, sha256=None, cache_dir=None):
    """Download a model file to the cache directory with optional SHA-256 verification.

    Returns the local path to the downloaded file, or None on failure.
    """
    import urllib.request

    dest_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "susurrus", "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    # Already cached and verified
    if os.path.isfile(dest_path):
        if sha256 is None or verify_sha256(dest_path, sha256):
            return dest_path
        logger.warning("Cached file %s failed SHA-256 check, re-downloading", filename)

    logger.info("Downloading %s from %s", filename, url)
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        logger.warning("Failed to download %s: %s", filename, e)
        return None

    if sha256 and not verify_sha256(dest_path, sha256):
        logger.error("Downloaded %s failed SHA-256 verification", filename)
        os.remove(dest_path)
        return None

    return dest_path


# --------------------------------------------------------------------------
# Companion model resolution (like CrisperWeaver's companions)
# --------------------------------------------------------------------------

# Maps TTS backends to their required companion models
COMPANION_MODELS = {
    "qwen3-tts": {
        "codec": {
            "name": "qwen3-tts-tokenizer-12hz",
            "flag": "--codec-model",
            "kwarg": "tts_codec_model",
        },
    },
    "orpheus": {
        "codec": {
            "name": "orpheus-snac-codec",
            "flag": "--codec-model",
            "kwarg": "tts_codec_model",
        },
    },
    "mimo-asr": {
        "codec": {
            "name": "mimo-tokenizer",
            "flag": "--codec-model",
            "kwarg": "tts_codec_model",
        },
    },
}


def resolve_companions(backend_name):
    """Return companion model metadata for a backend, or empty dict."""
    return COMPANION_MODELS.get(backend_name, {})


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------


def compute_metrics(audio_seconds, elapsed_seconds, word_count=0):
    """Compute performance metrics like CrisperWeaver.

    Returns a dict with:
        rtf: Real-Time Factor (audio_s / wall_s — higher = faster)
        wps: Words per second
        audio_s: Audio duration
        wall_s: Wall-clock time
    """
    rtf = audio_seconds / elapsed_seconds if elapsed_seconds > 0 else 0
    wps = word_count / elapsed_seconds if elapsed_seconds > 0 else 0
    return {
        "rtf": round(rtf, 2),
        "wps": round(wps, 1),
        "audio_s": round(audio_seconds, 1),
        "wall_s": round(elapsed_seconds, 1),
    }
