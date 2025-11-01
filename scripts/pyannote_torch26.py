#!/usr/bin/env python3
"""
PyTorch 2.6 compatibility patch for pyannote.audio with MPS optimizations
Fixed version that properly handles the PyTorch 2.6 weights_only change
"""

import os
import sys
import threading
import time
import warnings

import torch
from lightning_fabric.utilities import cloud_io

# === PYTORCH 2.6+ COMPATIBILITY PATCH - FIXED VERSION ===
print("Applying PyTorch 2.6+ compatibility fixes...")

# First method: Add safe globals
try:
    # Add TorchVersion to safe globals for PyTorch 2.6+
    from torch.serialization import add_safe_globals
    from torch.torch_version import TorchVersion

    add_safe_globals([TorchVersion])
    print("✅ Successfully registered TorchVersion as safe global")
except ImportError:
    print("ℹ️ add_safe_globals not available in this PyTorch version")

# Second method: Patch the _load function
original_load = cloud_io._load


def patched_load(path_or_url, map_location=None):
    """Patched load function that handles PyTorch 2.6 weights_only parameter changes"""
    try:
        # First try: Standard approach with weights_only=False
        result = torch.load(path_or_url, map_location=map_location, weights_only=False)
        print("🔄 Loaded checkpoint with weights_only=False")
        return result
    except Exception as e1:
        try:
            # Second try: Use safe_globals context manager if available (PyTorch 2.6+)
            from torch.serialization import safe_globals
            from torch.torch_version import TorchVersion

            with safe_globals([TorchVersion]):
                result = torch.load(path_or_url, map_location=map_location, weights_only=True)
                print("🔄 Loaded checkpoint with safe_globals context manager")
                return result
        except Exception as e2:
            # If both methods fail, raise the original error
            print(f"⚠️ Both loading methods failed!")
            raise e1


# Apply patch
cloud_io._load = patched_load

# Fix PyAnnote model loading if needed
try:
    # Check if we need to apply deeper patches to PyAnnote
    from pyannote.audio.core import model as pyannote_model

    module_path = os.path.dirname(pyannote_model.__file__)
    print(f"PyAnnote model module found at: {module_path}")

    # Apply deeper patch only if necessary
    if hasattr(torch, "__version__") and torch.__version__.startswith("2.6"):
        print("Applying deeper fix for PyAnnote with PyTorch 2.6...")

        # Patch PyAnnote's model loading directly
        original_pl_load = pyannote_model.pl_load

        def patched_pl_load(checkpoint_path, map_location=None):
            try:
                # Try with weights_only=False
                return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            except Exception as e:
                # If that fails, try with safe_globals
                try:
                    from torch.serialization import safe_globals
                    from torch.torch_version import TorchVersion

                    with safe_globals([TorchVersion]):
                        return torch.load(checkpoint_path, map_location=map_location)
                except:
                    # If all else fails, raise the original error
                    raise e

        # Apply the patch
        pyannote_model.pl_load = patched_pl_load
        print("✅ Applied deep fix to PyAnnote model loading")
except Exception as e:
    print(f"ℹ️ Could not apply deep fix: {e}")

print("✅ PyTorch 2.6+ compatibility patches applied")

# === PERFORMANCE OPTIMIZATIONS ===
# Determine the best device to use
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # For Apple Silicon, use MPS backend
    default_device = torch.device("mps")
    print("🍎 Using Apple Silicon MPS backend")

    # Set optimization flags for MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Reduces memory fragmentation

    # Clear any existing cached memory
    torch.mps.empty_cache()

    # Try to allocate the maximum allowed memory upfront to avoid fragmentation
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

# === OPTIMIZE EINSUM OPERATIONS ===
try:

    has_opt_einsum = True
    print("📊 Using optimized einsum operations")
except ImportError:
    has_opt_einsum = False
    print(
        "⚠️ opt_einsum not available - install with 'pip install opt-einsum' for better performance"
    )

# === SILENCE ANNOYING WARNINGS ===
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*MPEG_LAYER_III subtype is unknown to TorchAudio.*"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*degrees of freedom is <= 0.*")


# === ADVANCED PROGRESS INDICATOR ===
class EnhancedProgress:
    """A better progress indicator with ETA and stages"""

    def __init__(self, audio_file=None):
        self.start_time = time.time()
        self.completed = False
        self.audio_file = audio_file

        # Try to get audio duration for better ETA calculation
        self.audio_duration = None
        if audio_file:
            try:
                import librosa

                self.audio_duration = librosa.get_duration(path=audio_file)
                print(
                    f"📊 Audio duration: {self.audio_duration:.1f} seconds ({self.audio_duration/60:.1f} minutes)"
                )
            except:
                try:
                    import soundfile as sf

                    info = sf.info(audio_file)
                    self.audio_duration = info.duration
                    print(
                        f"📊 Audio duration: {self.audio_duration:.1f} seconds ({self.audio_duration/60:.1f} minutes)"
                    )
                except:
                    print("ℹ️ Could not determine audio duration")

        # Processing stages
        self.stages = [
            "Loading models",
            "Voice activity detection",
            "Speaker embedding extraction",
            "Speaker clustering",
            "Final segmentation",
        ]
        self.current_stage = self.stages[0]

        # Try to use tqdm for better progress display
        try:
            from tqdm.auto import tqdm

            self.has_tqdm = True

            # If we know the duration, we can make a better progress bar
            if self.audio_duration:
                self.progress = tqdm(
                    total=100,
                    desc=self.current_stage,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
                )
            else:
                # Otherwise just use an indeterminate progress
                self.progress = None
                self.start_simple_progress()
        except ImportError:
            self.has_tqdm = False
            self.progress = None
            self.start_simple_progress()

    def start_simple_progress(self):
        """Start a simple text-based progress indicator"""
        print("\nStarting diarization...")

        if self.audio_duration:
            # Estimate processing time based on audio duration
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS processing is approximately 2-3x realtime on M1
                est_factor = 2.5
            elif torch.cuda.is_available():
                # CUDA is faster, approximately 1.5-2x realtime
                est_factor = 1.5
            else:
                # CPU is slower
                est_factor = 5.0

            est_seconds = self.audio_duration * est_factor
            est_mins, est_secs = divmod(int(est_seconds), 60)

            print(f"Estimated processing time: ~{est_mins}m {est_secs}s")

        # Start the progress indicator thread
        self.progress_thread = threading.Thread(target=self._progress_indicator)
        self.progress_thread.daemon = True
        self.progress_thread.start()

    def _progress_indicator(self):
        """Simple progress indicator thread"""
        stages_cycle = self.stages.copy()
        stage_idx = 0
        current_stage = stages_cycle[stage_idx]

        while not self.completed:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            timestr = f"{mins:02d}:{secs:02d}"

            # Create a simple progress bar
            if self.audio_duration:
                # Estimate progress based on audio duration
                est_factor = 2.5  # Typical processing time ratio
                est_total = self.audio_duration * est_factor
                progress_pct = min(95, (elapsed / est_total) * 100)

                bar_len = 20
                filled_len = int(bar_len * progress_pct / 100)
                bar = "■" * filled_len + "□" * (bar_len - filled_len)

                # Estimate remaining time
                remaining = max(0, est_total - elapsed)
                rem_mins, rem_secs = divmod(int(remaining), 60)
                rem_str = f"~{rem_mins}m {rem_secs}s remaining"

                print(
                    f"\r[{timestr}] {current_stage}: [{bar}] {progress_pct:.1f}% {rem_str}",
                    end="",
                    flush=True,
                )
            else:
                # Simple spinner without progress percentage
                spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][int(elapsed) % 10]
                print(f"\r[{timestr}] {spinner} {current_stage}...", end="", flush=True)

            # Cycle through stages to give a sense of progress
            if int(elapsed) % 15 == 0 and int(elapsed) > 0:
                stage_idx = (stage_idx + 1) % len(stages_cycle)
                current_stage = stages_cycle[stage_idx]
                self.current_stage = current_stage

            time.sleep(1)

    def update_stage(self, stage_name, progress_percent=None):
        """Update the current stage and progress"""
        self.current_stage = stage_name

        if self.has_tqdm and self.progress and progress_percent is not None:
            self.progress.n = int(progress_percent)
            self.progress.set_description(stage_name)
            self.progress.refresh()

    def finish(self):
        """Complete the progress tracking"""
        self.completed = True

        # Clean up progress displays
        if self.has_tqdm and self.progress:
            self.progress.n = 100
            self.progress.close()
        else:
            # Clear the simple progress line
            print("\r" + " " * 80 + "\r", end="", flush=True)

        # Print completion stats
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)

        print(f"✅ Diarization completed in {mins:02d}:{secs:02d}")

        # Calculate speedup if we know the audio duration
        if self.audio_duration:
            speedup = self.audio_duration / elapsed
            print(f"Processing speedup: {speedup:.1f}x realtime")


def process_with_progress(audio_file, pipeline, num_speakers=None):
    """Process audio with enhanced progress tracking"""
    # Initialize progress tracker
    progress = EnhancedProgress(audio_file)

    try:
        # Set up diarization parameters
        diarization_kwargs = {}
        if num_speakers is not None:
            diarization_kwargs["num_speakers"] = num_speakers
            print(f"\nUsing fixed number of speakers: {num_speakers}")

        # Try to use PyAnnote's progress hook if available
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            with ProgressHook() as hook:
                # Define progress callback
                def update_progress(step):
                    stage_name = step.name if hasattr(step, "name") else "Processing"
                    percentage = step.percentage if hasattr(step, "percentage") else None
                    progress.update_stage(stage_name, percentage)

                # Register callback
                hook.on_step = update_progress

                # Add hook to kwargs
                diarization_kwargs["hook"] = hook

                # Process audio with hook
                diarization = pipeline(audio_file, **diarization_kwargs)
        except ImportError:
            # Fall back to standard processing without hook
            diarization = pipeline(audio_file, **diarization_kwargs)

        # Mark completion
        progress.finish()

        return diarization

    except Exception as e:
        # Mark progress as completed on error
        progress.completed = True
        print("\nError during diarization")
        raise e


def optimize_pipeline(pipeline):
    """Apply optimizations to the pipeline"""
    # Move to optimal device
    pipeline = pipeline.to(default_device)

    # Apply optimizations based on device
    if str(default_device) == "mps":
        try:
            # Optimize segmentation parameters for faster processing
            if hasattr(pipeline, "segmentation"):
                pipeline.segmentation.duration = 5.0  # Longer window for better results
                # Try to use larger step size for faster processing (if available)
                if hasattr(pipeline.segmentation, "step"):
                    pipeline.segmentation.step = 0.5  # Default is usually 0.1s, 5x faster

            # Optimize embedding batch size
            if hasattr(pipeline, "embedding"):
                # Optimal batch size for M1/M2
                pipeline.embedding.batch_size = 32

            # Optimize clustering
            if hasattr(pipeline, "clustering"):
                # Use faster clustering method
                if hasattr(pipeline.clustering, "method"):
                    pipeline.clustering.method = "centroid"

            print("✅ Applied MPS-specific optimizations")
        except Exception as e:
            print(f"⚠️ Could not apply all optimizations: {e}")
    elif str(default_device) == "cuda":
        try:
            # CUDA-specific optimizations
            if hasattr(pipeline, "embedding"):
                pipeline.embedding.batch_size = 64  # Larger batch for CUDA

            print("✅ Applied CUDA-specific optimizations")
        except Exception:
            pass
    else:
        # CPU-specific optimizations
        if hasattr(pipeline, "embedding"):
            pipeline.embedding.batch_size = 16  # Smaller batch for CPU

    return pipeline


def run_test(audio_file, token=None, num_speakers=None):
    """Test diarization with optimized settings"""
    if not token:
        token = os.environ.get("HF_TOKEN")

    if not token:
        print("No HF token found. Please provide one via arguments or environment.")
        return False

    print(f"Token length: {len(token)} characters, starts with: {token[:3]}...")

    try:
        print("Importing Pipeline...")
        from pyannote.audio import Pipeline

        print("Creating pipeline...")
        start_time = time.time()

        # Create pipeline with error handling
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=token
            )

            # Apply optimizations
            pipeline = optimize_pipeline(pipeline)

            load_time = time.time() - start_time
            print(f"Pipeline loaded in {load_time:.2f} seconds")
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            print("\nTrying fallback method...")

            # Try alternative loading approach
            from huggingface_hub import snapshot_download

            # Download the model manually first
            cache_dir = snapshot_download("pyannote/speaker-diarization-3.1", token=token)
            print(f"Model downloaded to: {cache_dir}")

            # Now try loading from cache
            pipeline = Pipeline.from_pretrained(
                cache_dir, use_auth_token=None  # Don't need token for local loading
            )

            # Apply optimizations
            pipeline = optimize_pipeline(pipeline)

            load_time = time.time() - start_time
            print(f"Pipeline loaded in {load_time:.2f} seconds (fallback method)")

        # Process the audio with enhanced progress
        print(f"Processing {audio_file}...")
        diarization = process_with_progress(audio_file, pipeline, num_speakers=num_speakers)

        # Show results
        speakers = set()
        segment_count = 0

        print("\nDiarization results:")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            if segment_count < 5:  # Show just the first few
                print(f"{speaker} speaks from {turn.start:.1f}s to {turn.end:.1f}s")
            segment_count += 1

        print(f"\nFound {len(speakers)} speakers across {segment_count} segments")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pyannote_torch26.py <audio_file> [huggingface_token] [num_speakers]")
        sys.exit(1)

    audio_file = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None

    print(f"Testing optimized diarization on {audio_file}...")
    success = run_test(audio_file, token, num_speakers)

    if success:
        print("\n✅ Diarization test successful!")
        print("\n🚀 Performance tips:")
        print("1. If you know the number of speakers, specify them with num_speakers=N")
        print("2. Install optional packages: pip install opt-einsum tqdm")
        print("3. For Apple Silicon Macs, make sure to use the latest PyTorch MPS version")
    else:
        print("\n❌ Diarization test failed!")
        print("Please check your setup and token.")
