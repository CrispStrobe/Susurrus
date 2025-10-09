# backends/diarization/progress.py
"""Progress indicators for diarization"""
import threading
import time

import torch


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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipeline = pipeline.to(device)

    # Apply optimizations based on device
    if str(device) == "mps":
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
    elif str(device) == "cuda":
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
