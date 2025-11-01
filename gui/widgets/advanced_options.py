# gui/widgets/advanced_options.py
"""Advanced options widget - aggregates all advanced settings"""

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from .collapsible_box import CollapsibleBox


class AdvancedOptionsBox(CollapsibleBox):
    """Contains all advanced transcription options"""

    def __init__(self, parent=None):
        super().__init__("Advanced Options", parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()

        # Proxy settings row (will be shown/hidden)
        self.proxy_row = self._create_proxy_row()
        layout.addWidget(self.proxy_row)

        # Backend and model selection
        self.backend_row = self._create_backend_row()
        layout.addLayout(self.backend_row)

        # Model, device, language row
        self.model_row = self._create_model_row()
        layout.addLayout(self.model_row)

        # Chunk length row
        self.chunk_row = self._create_chunk_row()
        layout.addWidget(self.chunk_row)

        # Output format row
        self.output_format_row = self._create_output_format_row()
        layout.addWidget(self.output_format_row)

        # Time trimming row
        self.misc_row = self._create_misc_row()
        layout.addLayout(self.misc_row)

        self.setContentLayout(layout)

    def _create_proxy_row(self):
        """Create proxy settings row"""
        widget = QWidget()
        layout = QHBoxLayout()

        self.proxy_url = QLineEdit()
        self.proxy_url.setPlaceholderText("Enter proxy URL if needed")
        self.proxy_username = QLineEdit()
        self.proxy_username.setPlaceholderText("Proxy username (optional)")
        self.proxy_password = QLineEdit()
        self.proxy_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.proxy_password.setPlaceholderText("Proxy password (optional)")

        layout.addWidget(QLabel("Proxy URL:"))
        layout.addWidget(self.proxy_url)
        layout.addWidget(QLabel("Username:"))
        layout.addWidget(self.proxy_username)
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.proxy_password)

        widget.setLayout(layout)
        widget.setVisible(False)
        return widget

    def _create_backend_row(self):
        """Create backend selection row"""
        layout = QHBoxLayout()
        self.backend_selection = QComboBox()
        layout.addWidget(QLabel("Backend:"))
        layout.addWidget(self.backend_selection)
        return layout

    def _create_model_row(self):
        """Create model, device, language row"""
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Model:"))
        self.model_id = QComboBox()
        self.model_id.setEditable(True)
        layout.addWidget(self.model_id)

        layout.addWidget(QLabel("Device:"))
        self.device_selection = QComboBox()
        self.device_selection.addItems(["Auto", "CPU", "GPU", "MPS"])
        layout.addWidget(self.device_selection)

        layout.addWidget(QLabel("Language:"))
        self.language = QLineEdit()
        self.language.setPlaceholderText("en")
        layout.addWidget(self.language)

        return layout

    def _create_chunk_row(self):
        """Create chunk length row"""
        widget = QWidget()
        layout = QHBoxLayout()

        self.max_chunk_length = QLineEdit()
        self.max_chunk_length.setPlaceholderText("Max Chunk Length (seconds, 0=No Chunking)")
        self.max_chunk_length.setText("0")

        layout.addWidget(QLabel("Max Chunk Length:"))
        layout.addWidget(self.max_chunk_length)

        widget.setLayout(layout)
        widget.setVisible(False)
        return widget

    def _create_output_format_row(self):
        """Create output format row"""
        widget = QWidget()
        layout = QHBoxLayout()

        self.output_format_selection = QComboBox()
        self.output_format_selection.addItems(["txt", "srt", "vtt"])

        layout.addWidget(QLabel("Output Format:"))
        layout.addWidget(self.output_format_selection)

        widget.setLayout(layout)
        widget.setVisible(False)
        return widget

    def _create_misc_row(self):
        """Create time trimming row"""
        layout = QHBoxLayout()

        self.start_time = QLineEdit()
        self.start_time.setPlaceholderText("Start Time (seconds)")
        self.end_time = QLineEdit()
        self.end_time.setPlaceholderText("End Time (seconds)")

        layout.addWidget(QLabel("Start Time (s):"))
        layout.addWidget(self.start_time)
        layout.addWidget(QLabel("End Time (s):"))
        layout.addWidget(self.end_time)

        return layout
