"""Dependencies status dialog"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from utils.i18n import t


class DependenciesDialog(QDialog):
    """Dialog to display the status of dependencies"""

    def __init__(self, dependencies, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("dialog.dependencies.title"))
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Title
        title = QLabel(f"<h2>{t('heading.dependency_status')}</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Info text
        info = QLabel(t("heading.dependencies_required"))
        layout.addWidget(info)

        # Create a table for dependencies
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Dependency", "Status", "Details"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        # Populate table
        table.setRowCount(len(dependencies))
        row = 0

        for name, info in dependencies.items():
            # Name
            name_item = QTableWidgetItem(name)
            table.setItem(row, 0, name_item)

            # Status
            if info["installed"]:
                status_text = "✓ Installed"
                if info.get("version"):
                    status_text += f" (v{info['version']})"
                status_item = QTableWidgetItem(status_text)
                status_item.setForeground(QColor("green"))
            else:
                if info["required"]:
                    status_text = "❌ Missing"
                    status_item = QTableWidgetItem(status_text)
                    status_item.setForeground(QColor("red"))
                else:
                    status_text = "⚠ Optional"
                    status_item = QTableWidgetItem(status_text)
                    status_item.setForeground(QColor("orange"))

            table.setItem(row, 1, status_item)

            # Details
            details_item = QTableWidgetItem(info["message"])
            table.setItem(row, 2, details_item)

            row += 1

        layout.addWidget(table)

        # Add installation instructions for missing dependencies
        if any(not info["installed"] and info["required"] for info in dependencies.values()):
            instructions_label = QLabel(f"<b>{t('heading.install_instructions')}</b>")
            layout.addWidget(instructions_label)

            instructions_text = QLabel(t("help.install_pip"))
            instructions_text.setTextFormat(Qt.TextFormat.RichText)
            instructions_text.setOpenExternalLinks(True)
            layout.addWidget(instructions_text)

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
