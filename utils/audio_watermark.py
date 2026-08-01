"""Neural audio watermarking via AudioSeal (EU AI Act Art. 50(2)).

The declarative RIFF marker in ``utils.ai_marking`` makes synthetic audio
machine-readable, but it is metadata: strip the chunk, or transcode to MP3,
and it is gone. A neural watermark is embedded in the samples themselves and
survives re-encoding, resampling and clipping.

AudioSeal (Meta, MIT-licensed) is an optional dependency because it pulls in
model weights on top of torch. When it is absent this module degrades to a
no-op and the declarative marker still carries the Art. 50(2) obligation —
Susurrus never emits unmarked audio, it just emits less robustly marked audio.

    pip install audioseal          # or: pip install 'susurrus[watermark]'

The CrispASR backends watermark inside the binary and do not use this module.
"""

import logging

logger = logging.getLogger(__name__)

#: AudioSeal model cards. The 16-bit variant carries a short payload we do
#: not use; presence of the watermark is the signal that matters here.
_GENERATOR_MODEL = "audioseal_wm_16bits"
_DETECTOR_MODEL = "audioseal_detector_16bits"

#: Detection probability at or above which audio counts as watermarked.
#: AudioSeal's own guidance treats 0.5 as the decision boundary.
DETECTION_THRESHOLD = 0.5

_generator = None
_detector = None
_unavailable = False


def _load(kind):
    """Lazily load an AudioSeal model. Returns None when unavailable."""
    global _generator, _detector, _unavailable

    if _unavailable:
        return None

    cached = _generator if kind == "generator" else _detector
    if cached is not None:
        return cached

    try:
        from audioseal import AudioSeal
    except ImportError:
        logger.debug("audioseal not installed; neural watermarking unavailable")
        _unavailable = True
        return None

    try:
        if kind == "generator":
            _generator = AudioSeal.load_generator(_GENERATOR_MODEL)
            return _generator
        _detector = AudioSeal.load_detector(_DETECTOR_MODEL)
        return _detector
    except Exception as e:
        # Model download failures, torch version mismatches, no network —
        # all mean "no neural watermark", none mean "crash the synthesis".
        logger.warning("Could not load AudioSeal %s: %s", kind, e)
        _unavailable = True
        return None


def is_available():
    """Return True if neural watermarking can run."""
    return _load("generator") is not None


def _read_wav(path):
    """Read a WAV as a (1, 1, samples) float tensor plus its sample rate."""
    import torch
    import torchaudio

    wav, sample_rate = torchaudio.load(path)
    if wav.dim() == 2:
        # AudioSeal expects a batch dimension: (batch, channels, samples).
        wav = wav.unsqueeze(0)
    return wav.to(torch.float32), sample_rate


def embed_watermark(wav_path):
    """Embed an AudioSeal watermark into a WAV file in place.

    Returns True if the watermark was applied, False if AudioSeal is
    unavailable or the file could not be processed.
    """
    generator = _load("generator")
    if generator is None:
        return False

    try:
        import torchaudio

        wav, sample_rate = _read_wav(wav_path)
        watermarked = generator.get_watermarked_audio(wav, sample_rate=sample_rate)
        # Drop the batch dimension torchaudio.save does not expect.
        torchaudio.save(wav_path, watermarked.squeeze(0).detach().cpu(), sample_rate)
    except Exception as e:
        logger.warning("Neural watermarking failed for %s: %s", wav_path, e)
        return False

    logger.info("AudioSeal watermark embedded: %s", wav_path)
    return True


def detect_watermark(wav_path):
    """Detect an AudioSeal watermark in a WAV file.

    Returns:
        dict with ``watermarked`` (bool) and ``confidence`` (float), or None
        if AudioSeal is unavailable or the file could not be processed.
    """
    detector = _load("detector")
    if detector is None:
        return None

    try:
        wav, sample_rate = _read_wav(wav_path)
        result, _message = detector.detect_watermark(wav, sample_rate=sample_rate)
        confidence = float(result)
    except Exception as e:
        logger.warning("Watermark detection failed for %s: %s", wav_path, e)
        return None

    return {
        "watermarked": confidence >= DETECTION_THRESHOLD,
        "confidence": confidence,
        "threshold": DETECTION_THRESHOLD,
    }


def _reset_cache_for_tests():
    """Clear the module-level model cache (test helper)."""
    global _generator, _detector, _unavailable
    _generator = None
    _detector = None
    _unavailable = False
