"""Light and dark theme definitions + speaker/confidence color utilities."""

# Speaker colors — 8 distinct colors that cycle
SPEAKER_COLORS = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#F44336",  # red
    "#00BCD4",  # cyan
    "#8BC34A",  # light green
    "#E91E63",  # pink
]


def speaker_color(index):
    """Get a speaker color by index (cycles through 8 colors)."""
    return SPEAKER_COLORS[index % len(SPEAKER_COLORS)]


def confidence_color(confidence):
    """Get a color for a confidence score.

    Returns:
        (hex_color, label) tuple.
    """
    if confidence is None:
        return "#888888", "unknown"
    if confidence >= 0.8:
        return "#4CAF50", "high"
    if confidence >= 0.6:
        return "#FF9800", "medium"
    return "#F44336", "low"


DARK_THEME = """
    QWidget {
        background-color: #1e1e1e;
        color: #cccccc;
        font-family: 'Segoe UI', 'Noto Sans', 'DejaVu Sans', sans-serif;
        font-size: 13px;
    }
    QMenuBar {
        background-color: #2d2d2d;
        color: #cccccc;
    }
    QMenuBar::item:selected {
        background-color: #3d3d3d;
    }
    QMenu {
        background-color: #2d2d2d;
        color: #cccccc;
        border: 1px solid #3d3d3d;
    }
    QMenu::item:selected {
        background-color: #3d3d3d;
    }
    QPushButton {
        background-color: #3d3d3d;
        color: #cccccc;
        border: 1px solid #555555;
        padding: 6px 12px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #4d4d4d;
    }
    QPushButton:pressed {
        background-color: #555555;
    }
    QPushButton:disabled {
        background-color: #2d2d2d;
        color: #666666;
    }
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {
        background-color: #2d2d2d;
        color: #cccccc;
        border: 1px solid #555555;
        padding: 4px;
        border-radius: 3px;
    }
    QTabWidget::pane {
        border: 1px solid #3d3d3d;
    }
    QTabBar::tab {
        background-color: #2d2d2d;
        color: #aaaaaa;
        padding: 8px 16px;
        border: 1px solid #3d3d3d;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #1e1e1e;
        color: #cccccc;
    }
    QProgressBar {
        border: 1px solid #555555;
        border-radius: 3px;
        text-align: center;
        color: #cccccc;
    }
    QProgressBar::chunk {
        background-color: #2196F3;
        border-radius: 2px;
    }
    QGroupBox {
        border: 1px solid #3d3d3d;
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 12px;
    }
    QGroupBox::title {
        color: #aaaaaa;
    }
    QCheckBox, QRadioButton {
        color: #cccccc;
    }
    QScrollBar:vertical {
        background-color: #1e1e1e;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background-color: #555555;
        border-radius: 5px;
        min-height: 20px;
    }
"""

LIGHT_THEME = """
    QWidget {
        background-color: #fafafa;
        color: #212121;
        font-family: 'Segoe UI', 'Noto Sans', 'DejaVu Sans', sans-serif;
        font-size: 13px;
    }
    QMenuBar {
        background-color: #f5f5f5;
        color: #212121;
    }
    QMenuBar::item:selected {
        background-color: #e0e0e0;
    }
    QMenu {
        background-color: #ffffff;
        color: #212121;
        border: 1px solid #e0e0e0;
    }
    QMenu::item:selected {
        background-color: #e3f2fd;
    }
    QPushButton {
        background-color: #e0e0e0;
        color: #212121;
        border: 1px solid #bdbdbd;
        padding: 6px 12px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #d5d5d5;
    }
    QPushButton:pressed {
        background-color: #bdbdbd;
    }
    QPushButton:disabled {
        background-color: #f5f5f5;
        color: #9e9e9e;
    }
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {
        background-color: #ffffff;
        color: #212121;
        border: 1px solid #bdbdbd;
        padding: 4px;
        border-radius: 3px;
    }
    QTabWidget::pane {
        border: 1px solid #e0e0e0;
    }
    QTabBar::tab {
        background-color: #f5f5f5;
        color: #757575;
        padding: 8px 16px;
        border: 1px solid #e0e0e0;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #fafafa;
        color: #212121;
    }
    QProgressBar {
        border: 1px solid #bdbdbd;
        border-radius: 3px;
        text-align: center;
        color: #212121;
    }
    QProgressBar::chunk {
        background-color: #2196F3;
        border-radius: 2px;
    }
    QGroupBox {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 12px;
    }
    QGroupBox::title {
        color: #757575;
    }
    QCheckBox, QRadioButton {
        color: #212121;
    }
    QScrollBar:vertical {
        background-color: #fafafa;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background-color: #bdbdbd;
        border-radius: 5px;
        min-height: 20px;
    }
"""

THEMES = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}
