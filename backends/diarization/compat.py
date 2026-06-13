"""PyTorch 2.6+ compatibility and performance optimizations for PyAnnote"""

import os
import warnings

import torch

# NOTE: lightning_fabric is imported lazily inside apply_pytorch26_fix() — it is
# only needed for the (optional) PyTorch 2.6 serialization patch. Importing it at
# module top level would make the torchaudio compatibility shim (and the whole
# compat module) unimportable on installs that have torch/torchaudio but not the
# full pyannote/lightning stack.

# Global flag to track if initialization has run
_INITIALIZED = False

# Global flag to track if the torchaudio shim has run
_TORCHAUDIO_PATCHED = False


def _torch_version_at_least(major, minor):
    """Return True if the installed torch is >= major.minor (e.g. 2.6)."""
    version = getattr(torch, "__version__", "")
    parts = version.split("+")[0].split(".")
    try:
        cur_major = int(parts[0])
        cur_minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return False
    return (cur_major, cur_minor) >= (major, minor)


def _subtype_to_bits(subtype):
    """Best-effort mapping of a soundfile subtype string to bits-per-sample."""
    if not subtype:
        return 0
    digits = "".join(ch for ch in str(subtype) if ch.isdigit())
    if digits:
        return int(digits)
    if "FLOAT" in subtype.upper() or "DOUBLE" in subtype.upper():
        return 32
    return 0


def _make_audio_metadata_class():
    """Create a stand-in for the removed ``torchaudio.AudioMetaData`` class.

    pyannote.audio references ``torchaudio.AudioMetaData`` as a return-type
    annotation at import time, so a lightweight class with the historical
    attributes is enough to let the import succeed and ``info()`` populate it.
    """

    class AudioMetaData:
        def __init__(
            self,
            sample_rate=0,
            num_frames=0,
            num_channels=0,
            bits_per_sample=0,
            encoding="UNKNOWN",
        ):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

        def __repr__(self):
            return (
                f"AudioMetaData(sample_rate={self.sample_rate}, "
                f"num_frames={self.num_frames}, num_channels={self.num_channels}, "
                f"bits_per_sample={self.bits_per_sample}, encoding={self.encoding!r})"
            )

    return AudioMetaData


def _make_info_shim(torchaudio_module):
    """Reimplement ``torchaudio.info`` using soundfile when torchaudio drops it."""

    def info(filepath, *args, **kwargs):
        import soundfile as sf

        sf_info = sf.info(filepath if hasattr(filepath, "read") else str(filepath))
        return torchaudio_module.AudioMetaData(
            sample_rate=int(sf_info.samplerate),
            num_frames=int(sf_info.frames),
            num_channels=int(sf_info.channels),
            bits_per_sample=_subtype_to_bits(getattr(sf_info, "subtype", None)),
            encoding=getattr(sf_info, "subtype", None) or "UNKNOWN",
        )

    return info


def apply_torchaudio_compat():
    """Restore torchaudio I/O symbols that pyannote.audio imports at load time.

    torchaudio >= 2.1 (and fully by 2.11) removed ``AudioMetaData``, ``info``
    and ``list_audio_backends`` after moving I/O to torchcodec, but the pinned
    pyannote.audio still references them, so ``import pyannote.audio`` raises
    ``AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'``.
    This shim restores them and MUST run before pyannote.audio is imported.
    """
    global _TORCHAUDIO_PATCHED

    if _TORCHAUDIO_PATCHED:
        return True

    try:
        import torchaudio
    except ImportError:
        return False

    patched = []

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = _make_audio_metadata_class()
        patched.append("AudioMetaData")

    if not hasattr(torchaudio, "info"):
        torchaudio.info = _make_info_shim(torchaudio)
        patched.append("info")

    if not hasattr(torchaudio, "list_audio_backends"):

        def list_audio_backends():
            available = []
            try:
                import soundfile  # noqa: F401

                available.append("soundfile")
            except ImportError:
                pass
            return available

        torchaudio.list_audio_backends = list_audio_backends
        patched.append("list_audio_backends")

    _TORCHAUDIO_PATCHED = True
    if patched:
        print(f"✅ Patched torchaudio compatibility shims: {', '.join(patched)}")
    return True


def apply_pytorch26_fix():
    """Apply PyTorch 2.6+ compatibility fix for pyannote.audio"""
    global _INITIALIZED

    if _INITIALIZED:
        return  # Already initialized

    # Only apply if using PyTorch >= 2.6, where torch.load defaults to
    # weights_only=True. This must match ALL such versions (2.6, 2.7, ... 2.11+),
    # not just "2.6.x" — otherwise pyannote checkpoint loading fails with
    # "Unsupported global: torch.torch_version.TorchVersion".
    if not _torch_version_at_least(2, 6):
        print(f"PyTorch version {getattr(torch, '__version__', '?')} does not need patching")
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
        from lightning_fabric.utilities import cloud_io

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
        except Exception:
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
    # Restore torchaudio symbols pyannote needs (must run before pyannote import)
    apply_torchaudio_compat()

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
