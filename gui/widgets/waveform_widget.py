# gui/widgets/waveform_widget.py
"""Simple audio waveform visualization widget."""

import logging
import struct

from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class WaveformWidget(QWidget):
    """Displays a waveform from PCM audio samples.

    Call load_wav() to load a WAV file, or set_samples() directly.
    Segment regions can be highlighted via set_segments().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []  # downsampled float32 peaks
        self._segments = []  # list of (start_frac, end_frac, color)
        self._playback_pos = -1.0  # 0.0-1.0 fraction, -1 = hidden
        self.setMinimumHeight(60)
        self.setMaximumHeight(100)

    def load_wav(self, wav_path):
        """Load and downsample a WAV file for display."""
        try:
            import wave

            with wave.open(wav_path, "rb") as wf:
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                raw = wf.readframes(n_frames)

            if sample_width == 2:
                fmt = f"<{n_frames * n_channels}h"
                samples = list(struct.unpack(fmt, raw))
                samples = [s / 32768.0 for s in samples]
            elif sample_width == 4:
                fmt = f"<{n_frames * n_channels}i"
                samples = list(struct.unpack(fmt, raw))
                samples = [s / 2147483648.0 for s in samples]
            else:
                samples = [0.0] * n_frames

            # Mono mix
            if n_channels > 1:
                mono = []
                for i in range(0, len(samples), n_channels):
                    mono.append(sum(samples[i : i + n_channels]) / n_channels)
                samples = mono

            self._downsample(samples)
            self.update()
        except Exception as e:
            logging.warning("Failed to load waveform: %s", e)
            self._samples = []

    def set_samples(self, samples):
        """Set raw float32 samples directly (will be downsampled)."""
        self._downsample(samples)
        self.update()

    def set_segments(self, segments, duration):
        """Set segment highlight regions.

        Args:
            segments: list of (start_sec, end_sec, color_hex)
            duration: total audio duration in seconds
        """
        if duration <= 0:
            self._segments = []
        else:
            self._segments = [(s / duration, e / duration, c) for s, e, c in segments]
        self.update()

    def set_playback_position(self, fraction):
        """Set playback position indicator (0.0-1.0, or -1 to hide)."""
        self._playback_pos = fraction
        self.update()

    def _downsample(self, samples, target_width=800):
        """Downsample to target_width peak values for display."""
        n = len(samples)
        if n == 0:
            self._samples = []
            return
        if n <= target_width:
            self._samples = [abs(s) for s in samples]
            return
        chunk = n // target_width
        peaks = []
        for i in range(target_width):
            start = i * chunk
            end = min(start + chunk, n)
            peak = max(abs(samples[j]) for j in range(start, end))
            peaks.append(peak)
        self._samples = peaks

    def paintEvent(self, event):
        if not self._samples:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        mid = h // 2
        n = len(self._samples)

        # Draw segment highlights
        for start_frac, end_frac, color in self._segments:
            x1 = int(start_frac * w)
            x2 = int(end_frac * w)
            painter.fillRect(x1, 0, x2 - x1, h, QColor(color + "30"))  # translucent

        # Draw waveform
        pen = QPen(QColor("#2196F3"), 1)
        painter.setPen(pen)
        for i, peak in enumerate(self._samples):
            x = int(i * w / n)
            amp = int(peak * mid * 0.9)
            painter.drawLine(x, mid - amp, x, mid + amp)

        # Draw playback position
        if 0.0 <= self._playback_pos <= 1.0:
            px = int(self._playback_pos * w)
            pen = QPen(QColor("#F44336"), 2)
            painter.setPen(pen)
            painter.drawLine(px, 0, px, h)

        painter.end()

    def clear(self):
        """Clear the waveform display."""
        self._samples = []
        self._segments = []
        self._playback_pos = -1.0
        self.update()
