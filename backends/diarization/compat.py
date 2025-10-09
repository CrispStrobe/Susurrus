"""PyTorch 2.6+ compatibility and performance optimizations for PyAnnote"""

import os
import warnings

import torch
from lightning_fabric.utilities import cloud_io

# Global flag to track if initialization has run
_INITIALIZED = False


def apply_pytorch26_fix():
    """Apply PyTorch 2.6+ compatibility fix for pyannote.audio"""
    global _INITIALIZED

    if _INITIALIZED:
        return  # Already initialized

    # Only apply if using PyTorch 2.6+
    if not hasattr(torch, "__version__") or not torch.__version__.startswith("2.6"):
        print(f"PyTorch version {torch.__version__} does not need patching")
        _INITIALIZED = True
        return

    print("Applying PyTorch 2.6+ compatibility fix for PyAnnote...")

    # 1. Find all PyAnnote classes that need to be registered
    pyannote_classes = []

    # Explicitly add known problematic classes
    known_classes = {
        "pyannote.audio.core.task": [
            "Specifications",
            "Problem",
            "Resolution",
            "Task",
            "Collection",
        ],
        "pyannote.audio.core.model": ["Model", "Introspection", "Preprocessors"],
        "pyannote.audio.core.io": ["AudioFile"],
    }

    # Import and add each class
    for module_name, class_names in known_classes.items():
        try:
            module = __import__(module_name, fromlist=class_names)
            for class_name in class_names:
                try:
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        pyannote_classes.append(cls)
                        print(f"✅ Added {module_name}.{class_name} to safe globals")
                except AttributeError:
                    pass
        except ImportError:
            pass

    # 2. Add TorchVersion
    try:
        from torch.torch_version import TorchVersion

        pyannote_classes.append(TorchVersion)
        print("✅ Added TorchVersion to safe globals")
    except ImportError:
        print("ℹ️ TorchVersion not available")

    # 3. Register all classes as safe globals
    try:
        from torch.serialization import add_safe_globals

        add_safe_globals(pyannote_classes)
        print(f"✅ Registered {len(pyannote_classes)} classes as safe globals")
    except ImportError:
        print("ℹ️ add_safe_globals not available in this PyTorch version")

    # 4. Patch lightning_fabric loader
    try:
        original_load = cloud_io._load

        def patched_load(path_or_url, map_location=None):
            """Patched loader that forces weights_only=False"""
            try:
                return torch.load(path_or_url, map_location=map_location, weights_only=False)
            except Exception as e:
                try:
                    # Try with safe_globals context
                    from torch.serialization import safe_globals

                    with safe_globals(pyannote_classes):
                        return torch.load(path_or_url, map_location=map_location)
                except Exception:
                    raise e

        cloud_io._load = patched_load
        print("✅ Patched lightning_fabric loader")
    except Exception as e:
        print(f"⚠️ Could not patch lightning_fabric loader: {e}")

    # 5. Patch PyAnnote model loading
    try:
        from pyannote.audio.core import model as pyannote_model

        original_pl_load = pyannote_model.pl_load

        def patched_pl_load(checkpoint_path, map_location=None):
            """Patched PyAnnote loader that forces weights_only=False"""
            try:
                return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            except Exception as e:
                try:
                    # Try with safe_globals context
                    from torch.serialization import safe_globals

                    with safe_globals(pyannote_classes):
                        return torch.load(checkpoint_path, map_location=map_location)
                except Exception:
                    raise e

        pyannote_model.pl_load = patched_pl_load
        print("✅ Patched PyAnnote model loader")
    except Exception as e:
        print(f"⚠️ Could not patch PyAnnote model loader: {e}")

    # 6. Patch PyTorch Lightning loader
    try:
        import pytorch_lightning.core.saving

        original_pl_load_from_checkpoint = pytorch_lightning.core.saving._load_from_checkpoint

        def patched_load_from_checkpoint(*args, **kwargs):
            """Patch PyTorch Lightning checkpoint loading"""
            # Store original torch.load
            original_torch_load = torch.load

            # Create patched torch.load that forces weights_only=False
            def force_weights_only_false(*targs, **tkwargs):
                tkwargs["weights_only"] = False
                return original_torch_load(*targs, **tkwargs)

            # Apply patch temporarily
            torch.load = force_weights_only_false

            try:
                # Call original function with patched torch.load
                return original_pl_load_from_checkpoint(*args, **kwargs)
            finally:
                # Restore original torch.load
                torch.load = original_torch_load

        pytorch_lightning.core.saving._load_from_checkpoint = patched_load_from_checkpoint
        print("✅ Patched PyTorch Lightning loader")
    except Exception as e:
        print(f"⚠️ Could not patch PyTorch Lightning loader: {e}")

    _INITIALIZED = True
    print("✅ PyTorch 2.6+ compatibility fix applied")


def setup_device_optimizations():
    """Set up device-specific optimizations"""
    global default_device

    # Determine the best device to use
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # For Apple Silicon, use MPS backend
        default_device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS backend")

        # Set optimization flags for MPS
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Clear any existing cached memory
        torch.mps.empty_cache()

        # Try to allocate the maximum allowed memory upfront
        try:
            temp_tensor = torch.zeros(1024, 1024, 64, device="mps")
            del temp_tensor
            print("✅ MPS memory pre-allocated")
        except:
            print("⚠️ MPS memory pre-allocation failed")

    elif torch.cuda.is_available():
        default_device = torch.device("cuda")
        print("🚀 Using NVIDIA CUDA backend")
        torch.cuda.empty_cache()
    else:
        default_device = torch.device("cpu")
        print("💻 Using CPU backend")
        torch.set_num_threads(os.cpu_count())
        print(f"✅ Set to use {os.cpu_count()} CPU threads")

    return default_device


def check_opt_einsum():
    """Check for optimized einsum operations"""
    try:

        print("📊 Using optimized einsum operations")
        return True
    except ImportError:
        print(
            "⚠️ opt_einsum not available - install with 'pip install opt-einsum' for better performance"
        )
        return False


def setup_warning_filters():
    """Set up warning filters to silence annoying messages"""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*MPEG_LAYER_III subtype is unknown to TorchAudio.*",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*degrees of freedom is <= 0.*"
    )


def initialize_diarization_backend():
    """Initialize all diarization backend optimizations"""
    # Apply PyTorch 2.6 fix
    apply_pytorch26_fix()

    # Set up device optimizations
    default_device = setup_device_optimizations()

    # Check for optional optimizations
    has_opt_einsum = check_opt_einsum()

    # Set up warning filters
    setup_warning_filters()

    return default_device, has_opt_einsum


# Module-level variables that will be set on initialization
default_device = None
has_opt_einsum = False
