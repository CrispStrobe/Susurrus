# gui/widgets/segment_list_widget.py
"""Segment list view — per-segment display with speaker colors, confidence, inline editing."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class SegmentRow(QWidget):
    """A single segment row with timestamp, speaker chip, text, and confidence."""

    text_edited = pyqtSignal(int, str)  # (index, new_text)

    def __init__(self, index, segment, speaker_display=None, parent=None):
        super().__init__(parent)
        self._index = index
        self._segment = segment
        self._setup_ui(segment, speaker_display)

    def _setup_ui(self, seg, speaker_display):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Timestamp
        if seg.start > 0 or seg.end > 0:
            ts = f"{seg.start:.1f}-{seg.end:.1f}"
            ts_label = QLabel(ts)
            ts_label.setFixedWidth(100)
            ts_label.setStyleSheet("color: #888888; font-size: 11px;")
            layout.addWidget(ts_label)

        # Speaker chip
        if speaker_display:
            from gui.themes import speaker_color

            color = speaker_color(hash(speaker_display) % 8)
            chip = QLabel(speaker_display)
            chip.setFixedWidth(80)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setStyleSheet(
                f"background-color: {color}; color: white; border-radius: 4px; "
                f"padding: 2px 6px; font-size: 11px;"
            )
            layout.addWidget(chip)

        # Text (editable on double-click)
        self._text_edit = QLineEdit(seg.text)
        self._text_edit.setReadOnly(True)
        self._text_edit.setFrame(False)
        self._text_edit.setStyleSheet("background: transparent;")
        self._text_edit.editingFinished.connect(self._on_edit_done)
        layout.addWidget(self._text_edit, stretch=1)

        # Confidence badge
        if seg.confidence is not None:
            from gui.themes import confidence_color

            color, label = confidence_color(seg.confidence)
            badge = QLabel(f"{seg.confidence:.0%}")
            badge.setFixedWidth(40)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
            badge.setToolTip(f"Confidence: {label}")
            layout.addWidget(badge)

        # Edited indicator
        if seg.edited:
            edited_label = QLabel("*")
            edited_label.setToolTip("Edited")
            edited_label.setStyleSheet("color: #FF9800; font-weight: bold;")
            layout.addWidget(edited_label)

    def mouseDoubleClickEvent(self, event):
        """Enable editing on double-click."""
        self._text_edit.setReadOnly(False)
        self._text_edit.setFrame(True)
        self._text_edit.setFocus()
        self._text_edit.selectAll()

    def _on_edit_done(self):
        """Called when editing is finished."""
        self._text_edit.setReadOnly(True)
        self._text_edit.setFrame(False)
        new_text = self._text_edit.text()
        if new_text != self._segment.text:
            self.text_edited.emit(self._index, new_text)


class SegmentListWidget(QWidget):
    """Scrollable list of segment rows with speaker colors and inline editing.

    Replaces QPlainTextEdit for structured transcription output.
    Falls back to plain text display if no structured segments.
    """

    segment_edited = pyqtSignal(int, str)  # (index, new_text)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scroll area for segment rows
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._layout.addStretch()
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

        # Plain text fallback (hidden by default)
        self._plain_text = QPlainTextEdit()
        self._plain_text.setReadOnly(True)
        self._plain_text.setVisible(False)
        outer.addWidget(self._plain_text)

    def set_segments(self, transcription_result):
        """Populate from a TranscriptionResult."""
        self._clear_rows()

        if not transcription_result or not transcription_result.segments:
            self._scroll.setVisible(False)
            self._plain_text.setVisible(True)
            return

        self._scroll.setVisible(True)
        self._plain_text.setVisible(False)

        for i, seg in enumerate(transcription_result.segments):
            speaker_display = transcription_result.display_speaker(seg.speaker)
            row = SegmentRow(i, seg, speaker_display=speaker_display)
            row.text_edited.connect(self._on_row_edited)
            self._layout.insertWidget(self._layout.count() - 1, row)  # before stretch
            self._rows.append(row)

    def add_segment(self, index, segment, speaker_display=None):
        """Add a single segment row (for streaming updates)."""
        self._scroll.setVisible(True)
        self._plain_text.setVisible(False)

        row = SegmentRow(index, segment, speaker_display=speaker_display)
        row.text_edited.connect(self._on_row_edited)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows.append(row)

        # Auto-scroll to bottom
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_plain_text(self, text):
        """Show plain text (fallback mode)."""
        self._clear_rows()
        self._scroll.setVisible(False)
        self._plain_text.setVisible(True)
        self._plain_text.setPlainText(text)

    def append_plain_text(self, text):
        """Append to plain text view."""
        if not self._plain_text.isVisible():
            self._scroll.setVisible(False)
            self._plain_text.setVisible(True)
        self._plain_text.appendPlainText(text)

    def clear(self):
        """Clear all content."""
        self._clear_rows()
        self._plain_text.clear()

    def to_plain_text(self):
        """Get all text as plain string."""
        if self._plain_text.isVisible():
            return self._plain_text.toPlainText()
        texts = []
        for row in self._rows:
            texts.append(row._text_edit.text())
        return "\n".join(texts)

    def _clear_rows(self):
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows = []

    def _on_row_edited(self, index, new_text):
        self.segment_edited.emit(index, new_text)
