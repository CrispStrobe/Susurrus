# gui/widgets/batch_panel.py
"""Batch processing panel — queue multiple files for sequential transcription."""

import logging
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class BatchPanel(QWidget):
    """Panel for batch transcription of multiple audio files."""

    # Emitted when batch processing starts/finishes
    batch_started = pyqtSignal()
    batch_finished = pyqtSignal()
    job_done = pyqtSignal(object)  # BatchJob

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Batch Queue"))
        header.addStretch()
        self.status_label = QLabel("0 files")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        # File list
        self.list_widget = QListWidget()
        self.list_widget.setAcceptDrops(True)
        layout.addWidget(self.list_widget)

        # Buttons
        btn_row = QHBoxLayout()

        self.add_button = QPushButton("Add Files...")
        self.add_button.clicked.connect(self._on_add_files)
        btn_row.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._on_remove)
        btn_row.addWidget(self.remove_button)

        self.start_button = QPushButton("Start Batch")
        self.start_button.setStyleSheet("background-color: #388E3C; color: white;")
        self.start_button.clicked.connect(self._on_start)
        btn_row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_button)

        self.clear_button = QPushButton("Clear Done")
        self.clear_button.clicked.connect(self._on_clear_done)
        btn_row.addWidget(self.clear_button)

        layout.addLayout(btn_row)

    def set_queue(self, queue):
        """Set the BatchQueue instance to use."""
        self._queue = queue

    def add_files(self, file_paths):
        """Add files to the batch queue."""
        if not self._queue:
            return
        for path in file_paths:
            if os.path.isfile(path):
                self._queue.add(path)
        self._refresh_list()

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.opus *.webm);;All Files (*)",
        )
        if files:
            self.add_files(files)

    def _on_remove(self):
        row = self.list_widget.currentRow()
        if row >= 0 and self._queue:
            self._queue.remove(row)
            self._refresh_list()

    def _on_start(self):
        if not self._queue or not self._queue.jobs:
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.batch_started.emit()
        self._queue.start()

    def _on_stop(self):
        if self._queue:
            self._queue.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_clear_done(self):
        if self._queue:
            self._queue.clear_completed()
            self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        if not self._queue:
            return
        for job in self._queue.jobs:
            icon = {"queued": "⏳", "running": "▶", "done": "✓", "error": "✗"}.get(
                job.status, "?"
            )
            name = os.path.basename(job.file_path)
            display = f"{icon}  {name}"
            if job.status == "error":
                display += f"  — {job.error_message[:50]}"
            elif job.status == "done":
                display += f"  ({job.elapsed:.1f}s)"
            item = QListWidgetItem(display)
            self.list_widget.addItem(item)

        summary = self._queue.summary
        total = sum(summary.values())
        done = summary.get("done", 0)
        self.status_label.setText(f"{done}/{total} done")

    def update_display(self):
        """Refresh the list display (call from a timer or signal)."""
        self._refresh_list()
        if self._queue and not self._queue.is_running:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
