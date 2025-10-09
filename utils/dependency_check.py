# utils/dependency_check.py:
"""Dependency checking utilities"""
import importlib
import logging
import os
import subprocess


def check_dependencies():
    """Check for required dependencies with safer import handling"""

    dependencies = {
        "PyTorch": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "torch",
            "message": "PyTorch is essential for audio processing and transcription",
        },
        "Transformers": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "transformers",
            "message": "Transformers is required for Whisper models",
        },
        "PyAnnote Audio": {
            "required": False,
            "installed": False,
            "version": None,
            "module": "pyannote.audio",
            "message": "Required for speaker diarization",
        },
        "Pydub": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "pydub",
            "message": "Required for audio file processing",
        },
        "Hugging Face Hub": {
            "required": False,
            "installed": False,
            "version": None,
            "module": "huggingface_hub",
            "message": "Required for model downloading",
        },
        "NumPy": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "numpy",
            "message": "Required for audio processing",
        },
    }

    # Check each dependency with safer approach
    for name, info in dependencies.items():
        try:
            # Use importlib for safer importing
            import importlib

            module = importlib.import_module(info["module"])
            info["installed"] = True

            # Get version if available
            if hasattr(module, "__version__"):
                info["version"] = module.__version__
            elif hasattr(module, "version"):
                info["version"] = module.version

            version_str = f" v{info['version']}" if info["version"] else ""
            logging.info(f"{name}{version_str} is available")

            # Special checks for specific modules
            if name == "PyTorch":
                # Safer CUDA check that won't crash
                try:
                    cuda_available = module.cuda.is_available()
                    logging.info(f"CUDA available: {cuda_available}")
                except:
                    logging.info("Could not check CUDA availability")

                # Safer MPS check that won't crash
                try:
                    if hasattr(module.backends, "mps"):
                        mps_available = module.backends.mps.is_available()
                        logging.info(f"MPS available: {mps_available}")
                except:
                    logging.info("Could not check MPS availability")

        except ImportError:
            info["installed"] = False
            if info["required"]:
                logging.warning(f"{name} not found. {info['message']}")
            else:
                logging.info(f"{name} not found. {info['message']} (optional)")
        except Exception as e:
            # Instead of allowing the exception to propagate, just mark as not installed
            # and log the error
            info["installed"] = False
            logging.warning(f"Error checking {name}: {str(e)}")
            if info["required"]:
                logging.warning(f"{name} might not be usable. {info['message']}")
            else:
                logging.info(f"{name} might not be usable. {info['message']} (optional)")

    # Check for pyannote.audio version specifically while avoiding the full import
    # which could cause the dependency conflict
    try:
        import importlib.metadata

        pyannote_version = importlib.metadata.version("pyannote.audio")
        dependencies["PyAnnote Audio"]["installed"] = True
        dependencies["PyAnnote Audio"]["version"] = pyannote_version
        logging.info(f"PyAnnote Audio v{pyannote_version} is installed")
    except:
        # Already handled in the main loop
        pass

    # Check for ffmpeg
    try:
        import subprocess

        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            version_str = result.stdout.split("\n")[0]
            logging.info(f"ffmpeg is installed: {version_str}")
            dependencies["ffmpeg"] = {
                "required": True,
                "installed": True,
                "version": version_str.split(" ")[2],
                "message": "Required for audio format conversion",
            }
        else:
            logging.warning("ffmpeg command failed. Audio format support may be limited.")
            dependencies["ffmpeg"] = {
                "required": True,
                "installed": False,
                "message": "Required for audio format conversion",
            }
    except FileNotFoundError:
        logging.warning("ffmpeg not found. Audio format support may be limited.")
        dependencies["ffmpeg"] = {
            "required": True,
            "installed": False,
            "message": "Required for audio format conversion",
        }
    except Exception as e:
        logging.warning(f"Error checking ffmpeg: {str(e)}")
        dependencies["ffmpeg"] = {
            "required": True,
            "installed": False,
            "message": "Required for audio format conversion",
        }

    # Check for HF_TOKEN in environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logging.info("Hugging Face token found in environment.")
        dependencies["HF_TOKEN"] = {
            "required": False,
            "installed": True,
            "message": "Required for speaker diarization",
        }
    else:
        logging.info(
            "No Hugging Face token found in environment. You'll need to provide one for speaker diarization."
        )
        dependencies["HF_TOKEN"] = {
            "required": False,
            "installed": False,
            "message": "Required for speaker diarization",
        }

    # Return the dependency dict for potential UI display
    return dependencies


def check_ffmpeg_installation():
    """Check if ffmpeg is properly installed and working."""
    try:
        import subprocess

        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            version_str = result.stdout.split("\n")[0]
            logging.info(f"ffmpeg is installed: {version_str}")
            return True
        else:
            logging.warning("ffmpeg command failed. Output formats may be limited.")
            return False
    except FileNotFoundError:
        logging.warning("ffmpeg not found. Some audio formats may not be supported.")
        return False
    except Exception as e:
        logging.warning(f"Error checking ffmpeg: {str(e)}")
        return False


def is_diarization_available():
    """Check if diarization functionality is available without importing pyannote.audio directly"""
    try:
        # Use importlib.util to avoid actually loading the module
        import importlib.util

        # Check if the module is available
        spec = importlib.util.find_spec("pyannote.audio")
        if spec is None:
            return False

        # Check for HF_TOKEN which is required
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking diarization availability: {str(e)}")
        return False


def check_developer_mode():
    if platform.system() == "Windows":
        try:
            import ctypes

            # Check if process has admin rights
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if not is_admin:
                logging.warning("Python is not running with administrator privileges")

            # Check Developer Mode
            try:
                import winreg

                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock",
                    0,
                    winreg.KEY_READ,
                )
                value, _ = winreg.QueryValueEx(key, "AllowDevelopmentWithoutDevMode")
                if value != 1:
                    logging.warning("Developer Mode is not enabled")
                    QMessageBox.warning(
                        None,
                        "Developer Mode Not Enabled",
                        "Please enable Developer Mode in Windows Settings:\n"
                        "1. Open Windows Settings\n"
                        "2. Navigate to Privacy & security > For developers\n"
                        "3. Enable 'Developer Mode'\n\n"
                        "This will improve cache performance.",
                    )
            except WindowsError as e:
                logging.warning(f"Could not check Developer Mode registry: {e}")

        except Exception as e:
            logging.warning(f"Could not check Developer Mode status: {e}")
