# gui/widgets/voxtral_settings.py
"""Voxtral settings widget"""

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from .collapsible_box import CollapsibleBox


class VoxtralSettingsBox(CollapsibleBox):
    """Collapsible box for Voxtral API settings"""

    def __init__(self, parent=None):
        super().__init__("Voxtral API Settings", parent)
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            "Voxtral is Mistral AI's speech recognition model.\n"
            "It supports 8 languages and offers both local and API-based inference."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(info_label)

        # API key input
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("Mistral API Key:"))
        self.mistral_api_key = QLineEdit()
        self.mistral_api_key.setPlaceholderText("Enter your Mistral AI API key (for voxtral-api)")
        self.mistral_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.mistral_api_key.setToolTip(
            "Required for voxtral-api backend.\n"
            "Get your API key from: https://console.mistral.ai/"
        )
        api_key_layout.addWidget(self.mistral_api_key)

        # Show/Hide password button
        self.show_key_button = QPushButton("👁")
        self.show_key_button.setMaximumWidth(30)
        self.show_key_button.setCheckable(True)
        self.show_key_button.clicked.connect(self.toggle_api_key_visibility)
        api_key_layout.addWidget(self.show_key_button)

        # API key help button
        self.api_key_help_button = QPushButton("?")
        self.api_key_help_button.setMaximumWidth(30)
        self.api_key_help_button.clicked.connect(self.show_api_key_help)
        api_key_layout.addWidget(self.api_key_help_button)

        layout.addLayout(api_key_layout)

        # Supported languages info
        languages_label = QLabel(
            "<b>Supported Languages:</b> English, French, Spanish, German, "
            "Italian, Portuguese, Polish, Dutch"
        )
        languages_label.setWordWrap(True)
        languages_label.setStyleSheet("font-size: 11px; color: #999999;")
        layout.addWidget(languages_label)

        # Performance note
        note_label = QLabel(
            "<b>Note:</b> voxtral-local requires transformers from GitHub. "
            "Run install_voxtral.sh/bat to set up."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("font-size: 11px; color: #ff9900;")
        layout.addWidget(note_label)

        # Add layout to content area
        self.setContentLayout(layout)

    def toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_button.isChecked():
            self.mistral_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_button.setText("🙈")
        else:
            self.mistral_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_button.setText("👁")

    def show_api_key_help(self):
        """Show help dialog for Mistral API key"""
        QMessageBox.information(
            self,
            "Mistral API Key Help",
            "<h3>Mistral AI API Key</h3>"
            "<p>The Mistral API key is required for the <b>voxtral-api</b> backend.</p>"
            "<h4>How to get your API key:</h4>"
            "<ol>"
            "<li>Create a free account at <a href='https://console.mistral.ai/'>console.mistral.ai</a></li>"
            "<li>Navigate to API Keys section</li>"
            "<li>Create a new API key</li>"
            "<li>Copy and paste it here</li>"
            "</ol>"
            "<h4>Alternatively:</h4>"
            "<p>You can set the <code>MISTRAL_API_KEY</code> environment variable:</p>"
            "<p><b>PowerShell:</b> <code>$env:MISTRAL_API_KEY = 'your-key'</code></p>"
            "<p><b>CMD:</b> <code>set MISTRAL_API_KEY=your-key</code></p>"
            "<p><b>Linux/Mac:</b> <code>export MISTRAL_API_KEY='your-key'</code></p>",
        )
