# gui/widgets/collapsible_box.py
"""Collapsible box widget"""

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton()
        self.toggle_button.setStyleSheet(self.get_button_style())
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle)
        self.title = title
        self.update_toggle_button_text()

        self.content_area = QWidget()
        self.content_area.setVisible(False)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)
        self.setLayout(self.main_layout)

    def get_button_style(self):
        return """
            text-align: left;
            background-color: #3a3a3a;
            color: white;
            padding: 5px;
            font-size: 14px;
            border: none;
        """

    def toggle(self):
        self.content_area.setVisible(self.toggle_button.isChecked())
        self.update_toggle_button_text()

    def update_toggle_button_text(self):
        arrow = "▼" if self.toggle_button.isChecked() else "►"
        self.toggle_button.setText(f"{arrow} {self.title}")

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
