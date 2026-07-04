# gui/widgets/log_viewer.py
"""Real-time log viewer with level filtering and search."""

import logging
from collections import deque

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Color per log level
_LEVEL_COLORS = {
    "DEBUG": "#888888",
    "INFO": "#cccccc",
    "WARNING": "#FF9800",
    "ERROR": "#F44336",
    "CRITICAL": "#D32F2F",
}


class LogRingBuffer(logging.Handler):
    """Logging handler that stores entries in a ring buffer."""

    def __init__(self, maxlen=2000):
        super().__init__()
        self.records = deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record):
        self.records.append(record)


class LogViewer(QWidget):
    """Real-time log viewer with level filtering and search."""

    def __init__(self, parent=None, ring_buffer=None):
        super().__init__(parent)
        self._buffer = ring_buffer or LogRingBuffer()
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)  # refresh every second

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Filter row
        filter_row = QHBoxLayout()

        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_filter.setCurrentText("INFO")
        self.level_filter.currentTextChanged.connect(self._refresh)
        filter_row.addWidget(self.level_filter)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs...")
        self.search_input.textChanged.connect(self._refresh)
        filter_row.addWidget(self.search_input)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear)
        filter_row.addWidget(self.clear_button)

        layout.addLayout(filter_row)

        # Log output
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(5000)
        layout.addWidget(self.output)

    def get_handler(self):
        """Return the logging handler to attach to the root logger."""
        return self._buffer

    def _refresh(self):
        level_text = self.level_filter.currentText()
        min_level = getattr(logging, level_text, 0) if level_text != "ALL" else 0
        search = self.search_input.text().lower()

        lines = []
        for record in self._buffer.records:
            if record.levelno < min_level:
                continue
            msg = self._buffer.format(record)
            if search and search not in msg.lower():
                continue
            lines.append(msg)

        # Only update if content changed
        new_text = "\n".join(lines[-500:])  # show last 500 matching
        if self.output.toPlainText() != new_text:
            self.output.setPlainText(new_text)
            # Scroll to bottom
            sb = self.output.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_clear(self):
        self._buffer.records.clear()
        self.output.clear()
