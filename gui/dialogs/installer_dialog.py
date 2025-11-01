# gui/dialogs/installer_dialog.py:
"""Installation dialogs for various dependencies"""
import subprocess
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout


class DependencyInstallThread(QThread):
    """Thread for installing dependencies"""

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, install_type="yt-dlp"):
        super().__init__()
        self.install_type = install_type

    def run(self):
        """Run the installation in a separate thread"""
        try:
            if self.install_type == "yt-dlp":
                self._install_yt_deps()
            elif self.install_type == "voxtral":
                self._install_voxtral_deps()
            elif self.install_type == "pydub":
                self._install_pydub()
            elif self.install_type == "diarization":
                self._install_diarization()
            elif self.install_type == "pytorch_cuda":
                self._install_pytorch_cuda()
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False, str(e))

    def _install_yt_deps(self):
        """Install yt-dlp downloader dependencies"""
        # Install yt-dlp
        self.progress_signal.emit("Installing yt-dlp...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], capture_output=True, text=True
        )
        if result.returncode != 0:
            self.progress_signal.emit(f"Error installing yt-dlp: {result.stderr}")
        else:
            self.progress_signal.emit("yt-dlp installed successfully")

        # Check ffmpeg
        self.progress_signal.emit("Checking for ffmpeg...")
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode != 0:
            self.progress_signal.emit(
                "ffmpeg not found. You need to install ffmpeg for audio conversion."
            )
            self.finished_signal.emit(False, "Dependencies installed but ffmpeg is missing")
        else:
            self.progress_signal.emit("ffmpeg found")
            self.finished_signal.emit(True, "All dependencies installed successfully")

    def _install_voxtral_deps(self):
        """Install Voxtral dependencies"""
        # Uninstall existing transformers
        self.progress_signal.emit("Uninstalling existing transformers...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "transformers"])

        # Install transformers from GitHub
        self.progress_signal.emit("Installing transformers from GitHub...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/huggingface/transformers.git",
            ],
            check=True,
        )

        # Install mistral-common with audio support
        self.progress_signal.emit("Installing mistral-common...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "mistral-common[audio]"], check=True
        )

        # Install soundfile
        self.progress_signal.emit("Installing soundfile...")
        subprocess.run([sys.executable, "-m", "pip", "install", "soundfile"], check=True)

        self.finished_signal.emit(True, "Voxtral dependencies installed successfully")

    def _install_pydub(self):
        """Install pydub"""
        self.progress_signal.emit("Installing pydub...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pydub"], capture_output=True, text=True
        )
        if result.returncode != 0:
            self.finished_signal.emit(False, f"Installation failed: {result.stderr}")
        else:
            self.finished_signal.emit(True, "pydub installed successfully")

    def _install_diarization(self):
        """Install diarization dependencies"""
        self.progress_signal.emit("Installing pyannote.audio...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyannote.audio", "huggingface_hub"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.finished_signal.emit(False, f"Installation failed: {result.stderr}")
        else:
            self.finished_signal.emit(True, "Diarization dependencies installed successfully")

    def _install_pytorch_cuda(self):
        """Install PyTorch with CUDA support"""
        # Uninstall current PyTorch
        self.progress_signal.emit("Uninstalling existing PyTorch...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]
        )

        # Install with CUDA support
        self.progress_signal.emit("Installing PyTorch with CUDA support...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.finished_signal.emit(False, f"Installation failed: {result.stderr}")
        else:
            self.finished_signal.emit(True, "PyTorch with CUDA installed successfully")


class InstallerDialog(QDialog):
    """Dialog for installing missing dependencies"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install Dependencies")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        title = QLabel("<h2>Install Missing Dependencies</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        info = QLabel("The following dependencies are needed for full functionality:")
        layout.addWidget(info)

        # FFMPEG button
        ffmpeg_btn = QPushButton("Install FFMPEG")
        ffmpeg_btn.clicked.connect(self.install_ffmpeg)
        layout.addWidget(ffmpeg_btn)

        # PyTorch with CUDA
        pytorch_btn = QPushButton("Install PyTorch with CUDA support")
        pytorch_btn.clicked.connect(self.install_pytorch_cuda)
        layout.addWidget(pytorch_btn)

        # Pydub button
        pydub_btn = QPushButton("Install pydub")
        pydub_btn.clicked.connect(self.install_pydub)
        layout.addWidget(pydub_btn)

        # Diarization dependencies button
        diarize_btn = QPushButton("Install Diarization Dependencies")
        diarize_btn.clicked.connect(self.install_diarization)
        layout.addWidget(diarize_btn)

        # yt-dlp dependencies button
        yt_btn = QPushButton("Install yt-delp Dependencies")
        yt_btn.clicked.connect(self.install_yt)
        layout.addWidget(yt_btn)

        # Voxtral dependencies button
        voxtral_btn = QPushButton("Install Voxtral Dependencies")
        voxtral_btn.clicked.connect(self.install_voxtral)
        layout.addWidget(voxtral_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def _run_installer_thread(self, install_type, title, description):
        """Generic method to run installer thread with progress dialog"""
        # Create progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle(title)
        progress_dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        info_label = QLabel(description)
        layout.addWidget(info_label)

        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)  # Indeterminate
        layout.addWidget(progress_bar)

        status_label = QLabel("Preparing...")
        layout.addWidget(status_label)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(progress_dialog.reject)
        layout.addWidget(cancel_button)

        progress_dialog.setLayout(layout)
        progress_dialog.show()

        # Create and start installation thread
        install_thread = DependencyInstallThread(install_type)

        def update_status(message):
            status_label.setText(message)

        def on_finished(success, message):
            progress_bar.setMaximum(100)
            progress_bar.setValue(100)
            cancel_button.setText("Close")

            if success:
                status_label.setText("Installation completed successfully!")
                QMessageBox.information(progress_dialog, "Success", message)
            else:
                status_label.setText(f"Installation completed with issues: {message}")
                QMessageBox.warning(progress_dialog, "Warning", message)

            progress_dialog.close()

        install_thread.progress_signal.connect(update_status)
        install_thread.finished_signal.connect(on_finished)
        install_thread.start()

        progress_dialog.exec()

    def install_ffmpeg(self):
        """Open FFMPEG download page with instructions"""
        import webbrowser

        webbrowser.open("https://www.gyan.dev/ffmpeg/builds/")

        QMessageBox.information(
            self,
            "FFMPEG Installation Instructions",
            "1. Download the 'essentials' build\n"
            "2. Extract the zip file to C:\\ffmpeg\n"
            "3. Add C:\\ffmpeg\\bin to your system PATH:\n"
            "   - Open Control Panel > System > Advanced System Settings\n"
            "   - Click 'Environment Variables'\n"
            "   - Edit the 'Path' variable and add C:\\ffmpeg\\bin\n"
            "   - Click OK and restart your terminal",
        )

    def install_pytorch_cuda(self):
        """Install PyTorch with CUDA support"""
        reply = QMessageBox.question(
            self,
            "Install PyTorch with CUDA",
            "This will reinstall PyTorch with CUDA support. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._run_installer_thread(
                "pytorch_cuda",
                "Installing PyTorch with CUDA",
                "Installing PyTorch with CUDA support...\n" "This may take several minutes.",
            )

    def install_pydub(self):
        """Install pydub"""
        self._run_installer_thread(
            "pydub", "Installing pydub", "Installing pydub for audio processing..."
        )

    def install_diarization(self):
        """Install diarization dependencies"""
        self._run_installer_thread(
            "diarization",
            "Installing Diarization Dependencies",
            "Installing pyannote.audio and huggingface_hub...\n"
            "You will still need to get a Hugging Face token.",
        )

    def install_yt(self):
        """Install yt-dlp downloader dependencies"""
        self._run_installer_thread(
            "yt-dlp",
            "Installing yt-dlp Dependencies",
            "Installing dependencies for yt-dlp downloading...\n"
        )

    def install_voxtral(self):
        """Install Voxtral dependencies"""
        reply = QMessageBox.question(
            self,
            "Install Voxtral Dependencies",
            "This will install the development version of transformers.\n\n"
            "The following packages will be installed:\n"
            "• transformers (from GitHub)\n"
            "• mistral-common[audio]\n"
            "• soundfile\n\n"
            "This may take several minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._run_installer_thread(
                "voxtral",
                "Installing Voxtral Dependencies",
                "Installing Voxtral dependencies...\n" "This may take several minutes.",
            )
