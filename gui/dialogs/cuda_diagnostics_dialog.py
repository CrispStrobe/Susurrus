# gui/dialogs/cuda_diagnostics_dialog.py
"""CUDA diagnostics dialog"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTextBrowser,
    QDialogButtonBox,
) 
from PyQt6.QtCore import Qt

from utils.device_detection import check_nvidia_installation


class CUDADiagnosticsDialog(QDialog):
    """Dialog showing detailed CUDA diagnostics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CUDA and NVIDIA Diagnostics")
        self.setMinimumSize(600, 400)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI with diagnostics"""
        """Show a dialog with detailed CUDA diagnostics"""

        # Get diagnostics
        diagnostics = check_nvidia_installation()

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("CUDA and NVIDIA Diagnostics")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>CUDA and NVIDIA Diagnostics</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Text browser for detailed output
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)

        # Format the diagnostics information
        html_content = "<style>table { width: 100%; border-collapse: collapse; }"
        html_content += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        html_content += "th { background-color: #f2f2f2; }</style>"

        # System info
        html_content += "<h3>System Information</h3>"
        html_content += "<table><tr><th>Component</th><th>Details</th></tr>"
        html_content += f"<tr><td>Operating System</td><td>{diagnostics['system']['os']}</td></tr>"
        html_content += (
            f"<tr><td>Python Version</td><td>{diagnostics['system']['python']}</td></tr>"
        )
        html_content += "</table><br>"

        # NVIDIA Driver
        html_content += "<h3>NVIDIA Driver</h3>"
        if diagnostics["nvidia_driver"]["detected"]:
            html_content += "<table><tr><th>Component</th><th>Details</th></tr>"
            html_content += f"<tr><td>Driver Detected</td><td>Yes</td></tr>"
            html_content += f"<tr><td>Driver Version</td><td>{diagnostics['nvidia_driver']['version']}</td></tr>"
            html_content += (
                f"<tr><td>GPU</td><td>{diagnostics['nvidia_driver']['details']}</td></tr>"
            )
            html_content += "</table>"
        else:
            html_content += "<p>No NVIDIA driver detected. <a href='https://www.nvidia.com/Download/index.aspx'>Download NVIDIA drivers</a></p>"

        # CUDA Toolkit
        html_content += "<h3>CUDA Toolkit</h3>"
        if diagnostics["cuda_toolkit"]["detected"]:
            html_content += "<table><tr><th>Component</th><th>Details</th></tr>"
            html_content += f"<tr><td>CUDA Detected</td><td>Yes</td></tr>"
            html_content += f"<tr><td>CUDA Version</td><td>{diagnostics['cuda_toolkit']['version'] or 'Unknown'}</td></tr>"
            html_content += (
                f"<tr><td>CUDA Path</td><td>{diagnostics['cuda_toolkit']['path']}</td></tr>"
            )
            html_content += "</table>"
        else:
            html_content += "<p>No CUDA Toolkit detected. <a href='https://developer.nvidia.com/cuda-downloads'>Download CUDA Toolkit</a></p>"

        # PyTorch
        html_content += "<h3>PyTorch</h3>"
        if "version" in diagnostics["pytorch"] and diagnostics["pytorch"]["version"]:
            html_content += "<table><tr><th>Component</th><th>Details</th></tr>"
            html_content += (
                f"<tr><td>PyTorch Version</td><td>{diagnostics['pytorch']['version']}</td></tr>"
            )

            if "+cu" in str(diagnostics["pytorch"]["version"]):
                cuda_version_from_pytorch = str(diagnostics["pytorch"]["version"]).split("+cu")[1]
                html_content += f"<tr><td>CUDA Support</td><td>Yes (built with CUDA {cuda_version_from_pytorch})</td></tr>"
            elif "+cpu" in str(diagnostics["pytorch"]["version"]):
                html_content += f"<tr><td>CUDA Support</td><td>No (CPU-only build)</td></tr>"

            html_content += f"<tr><td>torch.cuda.is_available()</td><td>{diagnostics['pytorch']['cuda_available']}</td></tr>"

            if diagnostics["pytorch"]["cuda_available"]:
                html_content += f"<tr><td>PyTorch CUDA Version</td><td>{diagnostics['pytorch']['cuda_version']}</td></tr>"
                html_content += f"<tr><td>CUDA Working</td><td>{diagnostics['pytorch'].get('cuda_working', 'Not tested')}</td></tr>"

                if "error" in diagnostics["pytorch"]:
                    html_content += (
                        f"<tr><td>CUDA Error</td><td>{diagnostics['pytorch']['error']}</td></tr>"
                    )

            html_content += "</table>"
        else:
            html_content += "<p>PyTorch not installed or not detected.</p>"

        # Recommendations
        html_content += "<h3>Recommendations</h3>"

        if not diagnostics["nvidia_driver"]["detected"]:
            html_content += "<p>• Install NVIDIA GPU drivers from the NVIDIA website</p>"

        if (
            not diagnostics["pytorch"]["cuda_available"]
            and diagnostics["nvidia_driver"]["detected"]
        ):
            html_content += "<p>• Reinstall PyTorch with CUDA support:</p>"
            html_content += "<pre>pip uninstall torch torchvision torchaudio\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121</pre>"

        if "cuda_working" in diagnostics["pytorch"] and not diagnostics["pytorch"]["cuda_working"]:
            html_content += (
                "<p>• Your CUDA setup has issues. Check the error message for details.</p>"
            )

        text_browser.setHtml(html_content)
        layout.addWidget(text_browser)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec()
