"""C2PA Content Credentials signing for audio files.

Uses the c2pa-audio library (https://github.com/CrispStrobe/c2pa-audio)
via ctypes Python bindings. Falls back gracefully if the library is not
installed.

This is used for Python-native TTS backends (edge-tts, piper, kokoro-onnx,
chatterbox, speecht5) that bypass the CrispASR binary and therefore don't
get C2PA signing automatically.

CrispASR-based TTS output is already signed by the binary itself.
"""

import logging

logger = logging.getLogger(__name__)

_c2pa = None
_c2pa_checked = False


def _get_c2pa():
    """Lazy-load the C2PA signer. Returns None if unavailable."""
    global _c2pa, _c2pa_checked
    if _c2pa_checked:
        return _c2pa
    _c2pa_checked = True
    try:
        # Try the standalone c2pa_audio Python package first
        from c2pa_audio import C2paAudio

        _c2pa = C2paAudio()
        logger.info("c2pa-audio library loaded (v%s)", _c2pa.version)
    except (ImportError, OSError) as e:
        logger.debug("c2pa-audio not available: %s", e)
        _c2pa = None
    return _c2pa


def is_available():
    """Check if C2PA signing is available."""
    return _get_c2pa() is not None


def sign_wav_file(wav_path, cert_pem=None, key_pem=None):
    """Sign a WAV file with C2PA Content Credentials in-place.

    Args:
        wav_path: Path to the WAV file to sign.
        cert_pem: Optional PEM certificate string. Uses bundled cert if None.
        key_pem: Optional PEM private key string. Uses bundled key if None.

    Returns:
        True if signed, False if c2pa-audio is not available.
    """
    c2pa = _get_c2pa()
    if c2pa is None:
        return False

    try:
        with open(wav_path, "rb") as f:
            wav_data = f.read()

        signed = c2pa.sign_wav(wav_data, cert_pem=cert_pem, key_pem=key_pem)

        with open(wav_path, "wb") as f:
            f.write(signed)

        logger.info("C2PA signed: %s", wav_path)
        return True
    except Exception as e:
        logger.warning("C2PA signing failed for %s: %s", wav_path, e)
        return False


def verify_wav_file(wav_path):
    """Verify C2PA Content Credentials in a WAV file.

    Args:
        wav_path: Path to the WAV file to verify.

    Returns:
        dict with keys: valid (bool), signature_valid, data_hash_valid,
        assertions_valid. Returns None if c2pa-audio is not available.
    """
    c2pa = _get_c2pa()
    if c2pa is None:
        return None

    try:
        with open(wav_path, "rb") as f:
            wav_data = f.read()

        result = c2pa.verify_wav(wav_data)
        return {
            "valid": result.valid,
            "signature_valid": result.signature_valid,
            "data_hash_valid": result.data_hash_valid,
            "assertions_valid": result.assertions_valid,
        }
    except Exception as e:
        logger.warning("C2PA verification failed for %s: %s", wav_path, e)
        return None
