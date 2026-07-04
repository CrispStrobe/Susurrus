# gui/widgets/history_panel.py
"""History browser panel — view, search, load, delete past transcriptions."""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HistoryPanel(QWidget):
    """Panel for browsing and managing transcription history."""

    # Emitted when user wants to load a history entry into the output
    load_entry_signal = pyqtSignal(object)  # HistoryEntry

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Search bar
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search history...")
        self.search_input.textChanged.connect(self._on_search)
        search_row.addWidget(self.search_input)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        search_row.addWidget(self.refresh_button)
        layout.addLayout(search_row)

        # Entry list
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # Action buttons
        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Load selected transcript into output")
        self.load_button.clicked.connect(self._on_load)
        button_row.addWidget(self.load_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete selected entry")
        self.delete_button.clicked.connect(self._on_delete)
        button_row.addWidget(self.delete_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.setToolTip("Delete all history entries")
        self.clear_button.clicked.connect(self._on_clear_all)
        button_row.addWidget(self.clear_button)

        button_row.addStretch()

        # Info label
        self.info_label = QLabel("")
        button_row.addWidget(self.info_label)

        layout.addLayout(button_row)

    def refresh(self):
        """Reload history entries from disk."""
        try:
            from utils.history_service import HistoryService

            svc = HistoryService()
            query = self.search_input.text().strip()
            if query:
                self._entries = svc.search(query)
            else:
                self._entries = svc.list_entries()
        except Exception as e:
            logging.warning("Failed to load history: %s", e)
            self._entries = []

        self._populate_list()

    def _on_search(self, text):
        self.refresh()

    def _populate_list(self):
        self.list_widget.clear()
        for entry in self._entries:
            title = entry.title
            meta_parts = [entry.created_at_str]
            if entry.backend:
                meta_parts.append(entry.backend)
            if entry.language:
                meta_parts.append(entry.language)
            seg_count = len(entry.segments)
            if seg_count:
                meta_parts.append(f"{seg_count} segments")
            meta = " | ".join(meta_parts)
            display = f"{title}\n  {meta}"
            item = QListWidgetItem(display)
            item.setData(256, entry.id)  # Qt.UserRole = 256
            self.list_widget.addItem(item)

        self.info_label.setText(f"{len(self._entries)} entries")

    def _selected_entry(self):
        item = self.list_widget.currentItem()
        if not item:
            return None
        entry_id = item.data(256)
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def _on_item_double_clicked(self, item):
        self._on_load()

    def _on_load(self):
        entry = self._selected_entry()
        if entry:
            self.load_entry_signal.emit(entry)

    def _on_delete(self):
        entry = self._selected_entry()
        if not entry:
            return
        reply = QMessageBox.question(
            self,
            "Delete Entry",
            f"Delete '{entry.title}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from utils.history_service import HistoryService

                HistoryService().delete(entry.id)
                self.refresh()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete: {e}")

    def _on_clear_all(self):
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Delete ALL history entries? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from utils.history_service import HistoryService

                HistoryService().clear_all()
                self.refresh()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to clear: {e}")
