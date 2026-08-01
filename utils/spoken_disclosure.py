"""Audible AI-disclosure prefix for Python-native TTS backends.

EU AI Act Art. 50(4) requires deployers to disclose that deepfake content is
artificially generated. Machine-readable marking (``utils.ai_marking``,
``utils.c2pa_signing``) addresses Art. 50(2), but a listener hears none of it.
The CrispASR binary prepends a spoken disclaimer of its own; this module gives
the Python-native backends (edge-tts, piper, kokoro-onnx, speecht5,
chatterbox) the same behaviour.

Approach: synthesize the disclosure phrase with the *same* backend and model,
then concatenate it in front of the content audio. Using the same backend
keeps the sample rate, channel count and bit depth aligned by construction —
no resampling, no format negotiation.
"""

import logging
import os
import wave

logger = logging.getLogger(__name__)


def disclosure_text(locale=None):
    """Return the spoken disclosure phrase for *locale* (default: current)."""
    from utils.i18n import t

    return t("disclosure.spoken", locale=locale)


def _wav_params(path):
    """Return (nchannels, sampwidth, framerate) for a WAV file, or None."""
    try:
        with wave.open(path, "rb") as w:
            return (w.getnchannels(), w.getsampwidth(), w.getframerate())
    except (wave.Error, OSError) as e:
        logger.warning("Could not read WAV params from %s: %s", path, e)
        return None


def concat_wavs(prefix_path, content_path, output_path):
    """Write *prefix_path* followed by *content_path* to *output_path*.

    Returns True on success. Returns False when the two files disagree on
    channel count, sample width or frame rate — emitting audibly corrupt audio
    would be worse than skipping the prefix, so the caller falls back to
    marking-only.
    """
    prefix_params = _wav_params(prefix_path)
    content_params = _wav_params(content_path)

    if prefix_params is None or content_params is None:
        return False

    if prefix_params != content_params:
        logger.warning(
            "Skipping spoken disclosure: format mismatch " "(disclosure %s vs content %s)",
            prefix_params,
            content_params,
        )
        return False

    try:
        with wave.open(prefix_path, "rb") as w:
            prefix_frames = w.readframes(w.getnframes())
        with wave.open(content_path, "rb") as w:
            content_frames = w.readframes(w.getnframes())

        with wave.open(output_path, "wb") as out:
            out.setnchannels(content_params[0])
            out.setsampwidth(content_params[1])
            out.setframerate(content_params[2])
            out.writeframes(prefix_frames + content_frames)
    except (wave.Error, OSError) as e:
        logger.warning("Could not write concatenated disclosure audio: %s", e)
        return False

    return True


def prepend_spoken_disclosure(backend, output_path, voice=None, locale=None):
    """Prepend an audible AI-disclosure to *output_path* in place.

    Args:
        backend: The TTS backend that produced the audio. Used to synthesize
            the disclosure in a matching format.
        output_path: The synthesized WAV to prefix.
        voice: Voice to speak the disclosure with. Defaults to the backend's
            own default rather than a cloned voice — a disclosure delivered in
            the cloned person's voice is the confusion we are trying to avoid.
        locale: Override the disclosure language.

    Returns:
        True if the disclosure was prepended, False otherwise.
    """
    if not output_path or not output_path.lower().endswith(".wav"):
        return False
    if not os.path.isfile(output_path):
        return False

    # Guard against recursion: synthesizing the disclosure must not itself
    # trigger a disclosure pass.
    if getattr(backend, "_synthesizing_disclosure", False):
        return False

    text = disclosure_text(locale)
    prefix_path = f"{output_path}.disclosure.tmp.wav"
    merged_path = f"{output_path}.merged.tmp.wav"

    backend._synthesizing_disclosure = True
    try:
        backend.synthesize(text, prefix_path, voice=voice)
    except Exception as e:
        # A backend that cannot speak the disclosure must not lose the user's
        # audio. Marking still applies; the caller reports the shortfall.
        logger.warning("Could not synthesize spoken disclosure: %s", e)
        _cleanup(prefix_path)
        return False
    finally:
        backend._synthesizing_disclosure = False

    if not os.path.isfile(prefix_path):
        logger.warning("Disclosure synthesis produced no audio")
        return False

    try:
        if not concat_wavs(prefix_path, output_path, merged_path):
            return False
        os.replace(merged_path, output_path)
    finally:
        _cleanup(prefix_path)
        _cleanup(merged_path)

    logger.info("Spoken AI disclosure prepended to %s", output_path)
    return True


def _cleanup(path):
    try:
        os.unlink(path)
    except OSError:
        pass
