# utils/device_detection.py
"""Device detection and CUDA diagnostics"""
import logging
import os
import platform
import subprocess

import torch


def check_cuda():
    """Enhanced CUDA detection with better diagnostics"""
    try:
        import torch

        # First check basic CUDA availability
        cuda_available = torch.cuda.is_available()
        logging.info(f"torch.cuda.is_available(): {cuda_available}")

        if not cuda_available:
            # If CUDA is not available, let's check why
            logging.warning("CUDA not initially detected. Running diagnostics...")

            # Check if NVIDIA driver is installed
            try:
                import subprocess

                nvidia_smi = subprocess.run(
                    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if nvidia_smi.returncode == 0:
                    logging.info("NVIDIA driver is installed, but PyTorch CUDA is not available.")
                    logging.info("This likely means PyTorch was installed without CUDA support.")
                    logging.info("PyTorch version: " + torch.__version__)
                    logging.info(
                        "Please reinstall PyTorch with CUDA support from https://pytorch.org/"
                    )

                    # Show a message box with instructions
                    from PyQt6.QtWidgets import QMessageBox

                    QMessageBox.warning(
                        None,
                        "CUDA Not Available",
                        "NVIDIA GPU detected, but PyTorch was installed without CUDA support.\n\n"
                        "Please reinstall PyTorch with CUDA support:\n"
                        "pip uninstall torch torchvision torchaudio\n"
                        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n\n"
                        "Then restart this application.",
                    )
                else:
                    logging.warning("NVIDIA driver not found. nvidia-smi failed.")
            except Exception as e:
                logging.warning(f"Could not check NVIDIA driver: {e}")

            # Check CUDA_PATH environment variable
            cuda_path = os.environ.get("CUDA_PATH", "")
            if cuda_path:
                logging.info(f"CUDA_PATH environment variable is set to: {cuda_path}")
            else:
                logging.warning("CUDA_PATH environment variable is not set")

            return False

        # If we get here, basic CUDA is available
        device_count = torch.cuda.device_count()
        logging.info(f"CUDA device count: {device_count}")

        if device_count == 0:
            logging.warning("CUDA is available but no devices found")
            return False

        # Get CUDA and driver details
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)

        logging.info(f"CUDA Version: {cuda_version}")
        logging.info(f"GPU: {device_name}")

        # Try a small CUDA operation to verify
        try:
            test_tensor = torch.tensor([1.0], device="cuda")
            test_result = test_tensor * 2
            logging.info(f"CUDA test successful: {test_result.cpu().item()} (expected: 2.0)")
            return True
        except Exception as e:
            logging.error(f"CUDA operation failed: {e}")
            return False

    except ImportError as e:
        logging.warning(f"PyTorch import failed: {e}")
        return False
    except Exception as e:
        logging.warning(f"CUDA check failed: {e}")
        return False


def check_nvidia_installation():
    """Comprehensive check of NVIDIA and CUDA installation"""
    import os
    import platform
    import subprocess
    import sys

    diagnostics = {
        "nvidia_driver": {"detected": False, "version": None, "details": None},
        "cuda_toolkit": {"detected": False, "version": None, "path": None},
        "pytorch": {"cuda_available": False, "version": None, "cuda_version": None},
        "system": {"os": platform.platform(), "python": sys.version},
    }

    # Check NVIDIA driver with nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            diagnostics["nvidia_driver"]["detected"] = True
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                diagnostics["nvidia_driver"]["version"] = parts[0]
                diagnostics["nvidia_driver"]["details"] = f"{parts[1]} ({parts[2]})"
            else:
                diagnostics["nvidia_driver"]["details"] = result.stdout.strip()
        else:
            logging.warning("nvidia-smi failed with error: " + result.stderr)
    except Exception as e:
        logging.warning(f"Failed to run nvidia-smi: {e}")

    # Check CUDA Toolkit installation
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        diagnostics["cuda_toolkit"]["path"] = cuda_path
        diagnostics["cuda_toolkit"]["detected"] = True

        # Try to determine CUDA version
        nvcc_path = os.path.join(
            cuda_path, "bin", "nvcc.exe" if platform.system() == "Windows" else "nvcc"
        )
        if os.path.exists(nvcc_path):
            try:
                result = subprocess.run(
                    [nvcc_path, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode == 0:
                    # Extract version from output like "Cuda compilation tools, release 11.8, V11.8.89"
                    for line in result.stdout.splitlines():
                        if "release" in line:
                            parts = line.split("release")
                            if len(parts) > 1:
                                version_part = parts[1].split(",")[0].strip()
                                diagnostics["cuda_toolkit"]["version"] = version_part
            except Exception as e:
                logging.warning(f"Failed to run nvcc: {e}")

    # Check PyTorch CUDA support
    try:
        import torch

        diagnostics["pytorch"]["version"] = torch.__version__

        if torch.cuda.is_available():
            diagnostics["pytorch"]["cuda_available"] = True
            diagnostics["pytorch"]["cuda_version"] = torch.version.cuda

            # Additional test to verify CUDA is working
            try:
                test_tensor = torch.tensor([1.0], device="cuda")
                test_result = test_tensor * 2
                if test_result.item() == 2.0:
                    diagnostics["pytorch"]["cuda_working"] = True
                else:
                    diagnostics["pytorch"]["cuda_working"] = False
            except Exception as e:
                diagnostics["pytorch"]["cuda_working"] = False
                diagnostics["pytorch"]["error"] = str(e)
    except ImportError:
        logging.warning("PyTorch not installed")
    except Exception as e:
        logging.warning(f"Error checking PyTorch: {e}")

    return diagnostics


def get_default_device():
    cuda_available = check_cuda()
    if cuda_available:
        return "GPU"
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS"
    except ImportError:
        pass
    return "CPU"


def diagnose_pytorch():
    import logging
    import platform
    import sys

    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {platform.platform()}")

    try:
        import torch

        logging.info(f"PyTorch version: {torch.__version__}")

        # Check CUDA availability
        logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

        # Check CUDA version PyTorch was built with
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

        # Get NVIDIA driver version
        if hasattr(torch.version, "cuda") and torch.cuda.is_available():
            logging.info(f"NVIDIA driver version: {torch.cuda.get_device_properties(0).name}")
        else:
            logging.info("No CUDA driver found")

        # Try importing CUDA toolkit
        try:

            logging.info("CUDA toolkit is installed")
        except ImportError:
            logging.info("CUDA toolkit not found in Python environment")

    except ImportError:
        logging.error("PyTorch is not installed")
        return False

    return True
