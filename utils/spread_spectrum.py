"""Dependency-free spread-spectrum audio watermark (EU AI Act Art. 50(2)).

AudioSeal in :mod:`utils.audio_watermark` is the stronger watermark, but it is
optional: it pulls in torch plus model weights, so a default Susurrus install
does not have it. Until now that left the declarative RIFF marker as the only
layer on a default install — and that marker is metadata, gone after any
transcode. Art. 50(2) asks for marking that is robust "as far as technically
feasible", and this costs nothing but numpy.

The scheme is byte-compatible with CrispASR's ``crispasr_watermark.h`` and
CrispTTS's ``watermark.py``: same key, same xoshiro128+ PRNG, same FFT size and
hop, same comb placement. Audio marked by any of the three is detectable by the
other two, which matters because a Susurrus user transcribing CrispASR output
should be able to tell it was AI-generated.

Placement follows CrispASR's ``wm_params``. The comb sits inside the speech band
(~1.5-4.8 kHz) rather than spread to 11.7 kHz: the wide version put ~20 of its
32 bins where clean TTS speech is near-silent, which was audible as a "tinny"
tone. ``CRISPASR_WATERMARK_LEGACY=1`` selects the old band for *embedding*, and
detection reads both so previously-marked audio still verifies — though the
legacy reading has to clear a stricter bar before it may override the primary
one, for the reason given at :data:`LEGACY_DETECTION_THRESHOLD`.

Detection correlates the comb against the averaged magnitude spectrum, weighted
by each bin's own deviation. An earlier version counted bare sign agreement over
the 32 bins and took the better of the two band placements. That threw away the
magnitudes and gave the noise two chances to win: measured over 1500 clips of
unwatermarked speech, harmonic stacks, tones and noise at 16/24/44.1 kHz and
0.5-15 s, it called **~12%** of them watermarked at its 0.65 threshold.
Weighting by deviation and gating the legacy band separately puts that at
**1.2%** while raising true detection from 93% to 98%. The numbers below are
from that run — rerun it before changing the threshold, and do not lower it on
the strength of one recording.

    unwatermarked (n=1500):            mean 0.52, p99 0.78, max 0.90
    watermarked, native rate (n=375):  mean 0.88, p1  0.76, min 0.70
    watermarked, 128k MP3 (same rate): mean 0.89, min 0.85

1.2% is a floor, not a triumph: a 32-bin comb read against one spectrum cannot
do much better, and the residual false positives are harmonic material whose
partials happen to land on comb bins. Treat a positive from this tier as
evidence, not proof — which is why nothing in the Art. 50 enforcement path is
allowed to depend on it (see ``apply_provenance`` on the CrispASR backends).

**A sibling statistic was ported, measured, and rejected — the data, so nobody
re-ports it blind.** CrispASR and CrispTTS both replaced this with a per-frame
one-sample t (sample count = frame count, not 32) plus a specificity check
against 15 decoy sign patterns. On *their* corpus — 1265 one-second clips of
real human speech — it reads 0.9% FP / 97.0% TP where the old sign statistic
read 5.2% / 68.6%, which is a large and real improvement.

It does not survive this project's corpus, which deliberately includes pure
tones and harmonic stacks. A stationary tone produces an enormous per-frame t
(median 49, max 1212), and with 15 decoys the MAD estimate is loose enough that
a third of them clear z >= 1. Measured over 288 unmarked clips across speech,
tones, formant stacks and noise at 16/24/44.1 kHz and 1-5 s:

    correlation (this module)     FP  0.00%   TP 79.2%
    frames t/z (ported)           FP 15.62%   TP 97.9%
    both required                 FP  0.00%   TP 79.2%
    correlation OR (t, z>=3)      FP  2.08%   TP 90.3%

Raising the decoy count to 31 or 63 moves it very little; the trade is real and
neither statistic dominates. This module keeps the correlation because the two
errors are not symmetric here: a false positive is an affirmative claim that
someone's real recording is AI-generated, while a miss is reported as "could
not check, not not-AI-generated" and cannot produce unmarked output — the
declarative floor is unconditional. The sibling statistic is the better choice
for a speech-only corpus at short durations, and if this module is ever
retargeted at one, that is the change to make.

**What it survives, and what it does not.** The mark rides on fixed *bin
indices* of a fixed-size FFT, so it is tied to the sample rate it was embedded
at. Transcoding survives — MP3, requantisation, interpolation loss — but
*resampling does not*: at 24k->16k the same bins are a different frequency band
and detection falls to chance (measured ~5%). Restoring the original rate
restores detection, which is why a 44.1k->16k->44.1k round trip verifies while
a plain 44.1k->16k does not. Compensating by sweeping candidate rate ratios was
tried and rejected: a max over ~12 hypotheses drove the false-positive rate to
100%. AudioSeal is the layer to use where resampling is expected.

**The mark is low-level, not inaudible.** This docstring claimed "~39.5 dB
measured on 20 s of real speech" — a figure from one recording, reported as
though it were the scheme's. Measured across 18 twenty-second segments at
16/24/44.1 kHz: **mean 18.9 dB, median 18.3 dB, worst 13.3 dB**, best 27.4 dB.
CrispTTS found the same overstatement in its own docs and landed on the same
range.

That distinction is load-bearing rather than pedantic. Inaudibility is the
argument for embedding a watermark in *every* output by default; at 13-19 dB on
sparse passages that argument does not hold, and an operator who needs the mark
to be genuinely imperceptible needs to know before shipping, not after someone
complains. The alpha is deliberately the band default (0.05) rather than the
wideband 0.08 — CrispTTS shipped 1.6x hotter than designed for a while by
letting a caller's stale default win, which cost 3-4 dB for nothing.

SNR is signal-dependent: the nudge scales with the mean bin magnitude, so peaky
material takes a proportionally louder mark than broadband speech, and passages
near silence take the worst of it. Use AudioSeal where imperceptibility
matters.

It is a fixed-key comb, so someone who knows the scheme can strip it deliberately
— AudioSeal remains the right choice where that matters.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

#: Shared across CrispASR, CrispTTS and Susurrus — do not change independently.
WATERMARK_KEY = 0x437269737041535F  # "CrispASR" in hex-ish
WATERMARK_NBINS = 32
FFT_SIZE = 1024
HOP = FFT_SIZE // 2

#: Correlation at or above which audio counts as watermarked. See the module
#: docstring for the measurements behind this value — at 0.65, where it used to
#: sit, ~12% of unwatermarked audio read as marked.
DETECTION_THRESHOLD = 0.78

#: The legacy band is a *second* hypothesis, and every extra hypothesis buys
#: the noise another chance to clear the bar. It therefore has to clear a
#: stricter one before it may override the primary reading: legacy-marked audio
#: is rare (the band changed because it was audible), so paying for it with a
#: higher false-positive rate on everything else is the wrong trade.
LEGACY_DETECTION_THRESHOLD = 0.85

_U64 = 0xFFFFFFFFFFFFFFFF


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _U64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return x, (z ^ (z >> 31)) & _U64


class _Prng:
    """xoshiro128+, matching ``crispasr_wm::prng`` exactly."""

    __slots__ = ("s0", "s1")

    def __init__(self, seed):
        # The C++ splitmix takes its argument by reference, so the second call
        # mutates s[0]: it ends up as the intermediate state, not the hash.
        _, s0_initial = _splitmix64(seed)
        self.s0, self.s1 = _splitmix64(s0_initial)

    def next(self):
        s0, s1 = self.s0, self.s1
        result = (s0 + s1) & _U64
        s1 ^= s0
        self.s0 = (((s0 << 55) | (s0 >> 9)) & _U64) ^ s1 ^ ((s1 << 14) & _U64)
        self.s1 = ((s1 << 36) | (s1 >> 28)) & _U64
        return result

    def next_u32(self, bound):
        return int(self.next() % bound)


def wm_params(n_fft, legacy=None):
    """Return ``(lo_bin, hi_bin, default_alpha)``, mirroring CrispASR."""
    if legacy is None:
        legacy = bool(os.environ.get("CRISPASR_WATERMARK_LEGACY"))
    lo_bin = n_fft // 16  # skip sub-bass; ~1.5 kHz @ 24 kHz
    if legacy:
        return lo_bin, n_fft // 2 - 1, 0.08  # ~11.7 kHz — audible comb
    return lo_bin, n_fft // 5, 0.05  # ~4.8 kHz — inside the speech band


def generate_bin_pattern(key, n_fft, n_bins, lo_bin=None, hi_bin=None):
    """Return ``(bin_index, sign)`` pairs for one comb placement."""
    if lo_bin is None or hi_bin is None:
        band_lo, band_hi, _ = wm_params(n_fft)
        lo_bin = band_lo if lo_bin is None else lo_bin
        hi_bin = band_hi if hi_bin is None else hi_bin
    rng = _Prng(key)
    span = hi_bin - lo_bin
    if span <= 0 or n_bins <= 0:
        return []
    bins = []
    for _ in range(n_bins):
        idx = lo_bin + rng.next_u32(span)
        sign = 1 if (rng.next() & 1) else -1
        bins.append((idx, sign))
    return bins


def embed(pcm, alpha=None):
    """Embed the watermark into float32 mono PCM, returning a new array.

    ``alpha`` of None or negative selects the band default; ``0`` is an
    explicit no-op that leaves the samples untouched.
    """
    n = len(pcm)
    if n < FFT_SIZE:
        return pcm.copy()

    lo_bin, hi_bin, default_alpha = wm_params(FFT_SIZE)
    if alpha is None or alpha < 0:
        alpha = default_alpha
    if alpha == 0.0:
        # The STFT round-trip is not bit-exact, so a zero-strength pass would
        # still perturb the audio while embedding nothing.
        return pcm.copy()

    bins = generate_bin_pattern(WATERMARK_KEY, FFT_SIZE, WATERMARK_NBINS, lo_bin, hi_bin)
    if not bins:
        return pcm.copy()

    window = np.hanning(FFT_SIZE).astype(np.float32)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    for start in range(0, n - FFT_SIZE + 1, HOP):
        frame = pcm[start : start + FFT_SIZE] * window
        spectrum = np.fft.rfft(frame)

        mags = np.abs(spectrum[1 : FFT_SIZE // 2])
        rms_mag = np.sqrt(np.mean(mags**2)) if len(mags) else 0.0
        nudge = alpha * rms_mag

        for b_idx, b_sign in bins:
            if b_idx >= len(spectrum):
                continue
            mag = abs(spectrum[b_idx])
            new_mag = max(mag + nudge * b_sign, 0.0)
            if mag > 1e-15:
                spectrum[b_idx] *= new_mag / mag
            elif b_sign > 0:
                spectrum[b_idx] = complex(nudge, 0.0)

        reconstructed = np.fft.irfft(spectrum, n=FFT_SIZE).astype(np.float32)
        out[start : start + FFT_SIZE] += reconstructed * window
        norm[start : start + FFT_SIZE] += window**2

    result = pcm.copy().astype(np.float64)
    mask = norm > 1e-8
    result[mask] = out[mask] / norm[mask]
    return result.astype(np.float32)


def detect(pcm):
    """Return detection confidence in [0, 1] for float32 mono PCM.

    Reads the primary band, and lets the legacy placement override only when it
    clears :data:`LEGACY_DETECTION_THRESHOLD` — so audio marked on the old band
    still verifies without the second hypothesis inflating false positives on
    everything else.
    """
    avg_mags = _average_spectrum(pcm)
    if avg_mags is None:
        return 0.0

    lo_bin, hi_bin, _ = wm_params(FFT_SIZE, legacy=False)
    primary = _correlate(avg_mags, lo_bin, hi_bin)

    lo_bin, hi_bin, _ = wm_params(FFT_SIZE, legacy=True)
    legacy = _correlate(avg_mags, lo_bin, hi_bin)

    return max(primary, legacy if legacy >= LEGACY_DETECTION_THRESHOLD else 0.0)


def _average_spectrum(pcm):
    """Return the frame-averaged magnitude spectrum, or None if too short.

    Averaging across frames before correlating is what makes this work on
    speech: per-frame noise cancels while the watermark, being identical in
    every frame, survives. Computed once and shared by both band readings —
    the FFTs dominate the cost and do not depend on comb placement.
    """
    n = len(pcm)
    if n < FFT_SIZE:
        return None

    window = np.hanning(FFT_SIZE).astype(np.float32)
    half = FFT_SIZE // 2

    all_mags = []
    for start in range(0, n - FFT_SIZE + 1, HOP):
        frame = pcm[start : start + FFT_SIZE] * window
        all_mags.append(np.abs(np.fft.rfft(frame)[:half]).astype(np.float64))
    if not all_mags:
        return None
    return np.mean(all_mags, axis=0)


def _correlate(avg_mags, lo_bin, hi_bin):
    """Correlate one comb placement against an averaged magnitude spectrum.

    Each bin contributes its *deviation* from the local mean, not merely the
    sign of that deviation. A bin sitting far above its neighbours is strong
    evidence; one a hair above them is nearly none, and counting both as a full
    vote is what let unwatermarked audio reach the old threshold on a lucky run
    of 21-of-32 coin flips. Weighting by |delta| and normalising by the total
    weight keeps the result in [0, 1] with 0.5 as "no evidence either way".

    Deltas are clamped to 1.0 so a single spectral spike — a tone landing on a
    comb bin — cannot dominate the other 31.
    """
    bins = generate_bin_pattern(WATERMARK_KEY, FFT_SIZE, WATERMARK_NBINS, lo_bin, hi_bin)
    if not bins:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for b_idx, b_sign in bins:
        if b_idx < 1 or b_idx >= len(avg_mags):
            continue
        neighbours = [
            avg_mags[b_idx + d] for d in range(-2, 3) if d != 0 and 1 <= b_idx + d < len(avg_mags)
        ]
        if not neighbours:
            continue
        local_mean = sum(neighbours) / len(neighbours)
        if local_mean < 1e-12 and avg_mags[b_idx] < 1e-12:
            continue
        delta = (avg_mags[b_idx] - local_mean) / max(local_mean, 1e-12)
        delta = max(-1.0, min(1.0, delta))
        weighted_sum += delta * b_sign
        total_weight += abs(delta)

    if total_weight < 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, (weighted_sum / total_weight + 1.0) / 2.0)))
