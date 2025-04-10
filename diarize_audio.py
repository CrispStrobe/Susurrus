#!/usr/bin/env python3
# diarize_audio.py - Speaker diarization module for Susurrus

import os
import sys
import logging
import tempfile
import time
import json
import threading
import warnings
from pathlib import Path
import torch
import numpy as np
from pydub import AudioSegment
from huggingface_hub import login, hf_hub_download, HfApi
from pyannote.audio import Pipeline

from lightning_fabric.utilities import cloud_io


# === PYTORCH 2.6+ COMPATIBILITY FIX ===
# This is a comprehensive fix for loading PyAnnote models with PyTorch 2.6+
def apply_pytorch26_fix():
    """Apply PyTorch 2.6+ compatibility fix for pyannote.audio"""
    
    # Only apply if using PyTorch 2.6+
    if not hasattr(torch, "__version__") or not torch.__version__.startswith("2.6"):
        print(f"PyTorch version {torch.__version__} does not need patching")
        return
    
    print("Applying PyTorch 2.6+ compatibility fix for PyAnnote...")
    
    # 1. Find all PyAnnote classes that need to be registered
    pyannote_classes = []
    
    # Explicitly add known problematic classes
    known_classes = {
        "pyannote.audio.core.task": ["Specifications", "Problem", "Resolution", "Task", "Collection"],
        "pyannote.audio.core.model": ["Model", "Introspection", "Preprocessors"],
        "pyannote.audio.core.io": ["AudioFile"]
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
    
    print("✅ PyTorch 2.6+ compatibility fix applied")

# Apply the fix
apply_pytorch26_fix()

# === PERFORMANCE OPTIMIZATIONS ===
# Determine the best device to use
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # For Apple Silicon, use MPS backend
    default_device = torch.device('mps')
    print("🍎 Using Apple Silicon MPS backend")
    
    # Set optimization flags for MPS
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduces memory fragmentation
    
    # Clear any existing cached memory
    torch.mps.empty_cache()
    
    # Try to allocate the maximum allowed memory upfront to avoid fragmentation
    try:
        temp_tensor = torch.zeros(1024, 1024, 64, device='mps')
        del temp_tensor
        print("✅ MPS memory pre-allocated")
    except:
        print("⚠️ MPS memory pre-allocation failed")
        
elif torch.cuda.is_available():
    default_device = torch.device('cuda')
    print("🚀 Using NVIDIA CUDA backend")
    torch.cuda.empty_cache()
else:
    default_device = torch.device('cpu')
    print("💻 Using CPU backend")
    torch.set_num_threads(os.cpu_count())
    print(f"✅ Set to use {os.cpu_count()} CPU threads")

# === OPTIMIZE EINSUM OPERATIONS ===
try:
    import opt_einsum
    has_opt_einsum = True
    print("📊 Using optimized einsum operations")
except ImportError:
    has_opt_einsum = False
    print("⚠️ opt_einsum not available - install with 'pip install opt-einsum' for better performance")

# === SILENCE ANNOYING WARNINGS ===
warnings.filterwarnings("ignore", category=UserWarning, 
                        message=".*MPEG_LAYER_III subtype is unknown to TorchAudio.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*degrees of freedom is <= 0.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

# === ENHANCED PROGRESS INDICATOR ===
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
                print(f"📊 Audio duration: {self.audio_duration:.1f} seconds ({self.audio_duration/60:.1f} minutes)")
            except:
                try:
                    import soundfile as sf
                    info = sf.info(audio_file)
                    self.audio_duration = info.duration
                    print(f"📊 Audio duration: {self.audio_duration:.1f} seconds ({self.audio_duration/60:.1f} minutes)")
                except:
                    print("ℹ️ Could not determine audio duration")
        
        # Processing stages
        self.stages = [
            "Loading models",
            "Voice activity detection",
            "Speaker embedding extraction",
            "Speaker clustering",
            "Final segmentation"
        ]
        self.current_stage = self.stages[0]
        
        # Try to use tqdm for better progress display
        try:
            from tqdm.auto import tqdm
            self.has_tqdm = True
            
            # If we know the duration, we can make a better progress bar
            if self.audio_duration:
                self.progress = tqdm(total=100, desc=self.current_stage, 
                                    bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
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
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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
                bar = '■' * filled_len + '□' * (bar_len - filled_len)
                
                # Estimate remaining time
                remaining = max(0, est_total - elapsed)
                rem_mins, rem_secs = divmod(int(remaining), 60)
                rem_str = f"~{rem_mins}m {rem_secs}s remaining"
                
                print(f"\r[{timestr}] {current_stage}: [{bar}] {progress_pct:.1f}% {rem_str}", 
                      end="", flush=True)
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

class DiarizationManager:
    """Manager class for speaker diarization using pyannote.audio"""

    # Define the available diarization models
    AVAILABLE_MODELS = {
        "Default": "pyannote/speaker-diarization-3.1",
        "Legacy": "pyannote/speaker-diarization@2.1",
    }
    
    def __init__(self, hf_token=None, device=None, model_name="Default"):
        """Initialize diarization manager
        
        Args:
            hf_token: Hugging Face API token
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
            model_name: Name of the diarization model to use
        """
        # Store token
        self.hf_token = hf_token
        
        # Set model name
        self.model_name = model_name
        if model_name not in self.AVAILABLE_MODELS:
            logging.warning(f"Unknown model name: {model_name}. Falling back to Default model.")
            self.model_name = "Default"
        
        # Set the device (cuda, mps, or cpu)
        self.device = self._detect_device() if device is None else device
        
        # Validate requested device
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            logging.warning("MPS requested but not available. Falling back to CPU.")
            self.device = "cpu"
            
        self.pipeline = None
        logging.info(f"Diarization will use device: {self.device}")
        logging.info(f"Selected diarization model: {self.model_name}")

    def _detect_device(self):
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logging.info("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = "cpu"
            logging.info("No GPU detected, using CPU")
        return device
    
    def get_model_id(self):
        """Get the model ID for the selected model name"""
        return self.AVAILABLE_MODELS.get(self.model_name, self.AVAILABLE_MODELS["Default"])
    
    @classmethod
    def list_available_models(cls):
        """Return a list of available models"""
        return list(cls.AVAILABLE_MODELS.keys())
    
    def _get_token(self):
        """Get the Hugging Face token from various sources"""
        if self.hf_token is not None:
            return self.hf_token
            
        # Try environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            self.hf_token = token
            return token
            
        # Try token file
        try:
            token_path = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    token = f.read().strip()
                    if token:
                        self.hf_token = token
                        return token
        except:
            pass
            
        return None
    
    def test_authentication(self):
        """Verify authentication with Hugging Face Hub using the test script approach"""
        token = self._get_token()
        
        if not token:
            logging.error("No Hugging Face token found. Please provide a token.")
            return False
        
        api = HfApi()
        models_to_check = {
            "pyannote/segmentation": "config.yaml",
            "pyannote/speaker-diarization": "config.yaml",
            "pyannote/speaker-diarization-3.1": "config.yaml",
        }
        
        all_access = True
        
        for model, test_file in models_to_check.items():
            try:
                info = api.model_info(model, token=token)
                logging.info(f"Can access model info for {model}")
                
                # Check if the file exists in the list of siblings
                available_files = [s.rfilename for s in info.siblings]
                if test_file not in available_files:
                    # Try to find any config file
                    config_files = [f for f in available_files if f.startswith("config")]
                    if config_files:
                        test_file = config_files[0]
                        logging.info(f"Using alternative file: {test_file}")
                    else:
                        logging.warning(f"File '{test_file}' not found in model '{model}'")
                        all_access = False
                        continue
                
                # Try downloading the file
                try:
                    hf_hub_download(model, test_file, token=token, local_files_only=False)
                    logging.info(f"Successfully downloaded '{test_file}' from {model}")
                except Exception as e:
                    logging.error(f"Failed to download '{test_file}' from {model}: {e}")
                    all_access = False

            except Exception as e:
                logging.error(f"Cannot access model {model}: {e}")
                all_access = False
        
        return all_access
    
    def initialize_pipeline(self):
        """Initialize the diarization pipeline with optimizations"""
        if self.pipeline is not None:
            return True
                
        # Get the token
        token = self._get_token()
        if not token:
            raise ValueError(
                "No Hugging Face token found. Please provide a valid token."
            )
        
        # Get the model to use
        model_id = self.get_model_id()
        logging.info(f"Loading speaker diarization pipeline '{model_id}'...")
        
        try:
            # Load the pipeline directly
            pipeline = Pipeline.from_pretrained(
                model_id,
                use_auth_token=token
            )
            
            # Move to MPS device
            pipeline = pipeline.to(torch.device('mps'))
            
            # Apply optimizations similar to working script
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Optimize embedding batch size
                    if hasattr(pipeline, 'embedding'):
                        # Optimal batch size for M1/M2
                        pipeline.embedding.batch_size = 32
                    
                    # Optimize clustering
                    if hasattr(pipeline, 'clustering'):
                        if hasattr(pipeline.clustering, 'method'):
                            pipeline.clustering.method = 'centroid'
                    
                    logging.info("Applied MPS-specific optimizations")
                except Exception as e:
                    logging.warning(f"Could not apply all optimizations: {e}")
            
            self.pipeline = pipeline
            logging.info("Speaker diarization pipeline loaded successfully")
            return True
                    
        except Exception as e:
            error_msg = str(e)
            
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                logging.error("Authentication failed. Please check your token.")
                raise ValueError("Invalid token or authentication failed.")
                    
            elif "403" in error_msg or "cannot access gated repo" in error_msg.lower():
                logging.error(f"You need to accept the model license on Hugging Face Hub.")
                raise ValueError(
                    f"Please visit https://huggingface.co/{model_id.split('@')[0]} "
                    f"to accept the license agreement."
                )
                    
            elif "found in model" in error_msg.lower() or "not found" in error_msg.lower():
                logging.error(f"Required files not found in the model repository.")
                if "Default" in self.model_name and "@" not in model_id:
                    # Try legacy model instead
                    logging.info("Trying legacy model instead...")
                    self.model_name = "Legacy"
                    return self.initialize_pipeline()
                        
            else:
                logging.error(f"Failed to load model: {e}")
                raise
    
    def diarize(self, audio_path, min_speakers=None, max_speakers=None):
        """Perform speaker diarization on an audio file using the approach from the working script"""
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Initialize the pipeline
        self.initialize_pipeline()
        
        try:
            logging.info(f"Processing {audio_path}...")
            
            # Convert min/max speakers to num_speakers parameter
            num_speakers = max_speakers
            if num_speakers is not None:
                logging.info(f"Using fixed number of speakers: {num_speakers}")
            
            # Use the same process_with_progress approach that works in the script
            diarization = process_with_progress(
                audio_path, self.pipeline, num_speakers=num_speakers
            )
            
            # Extract segments from the diarization result
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker.replace("SPEAKER_", "Speaker "),
                    "start": float(turn.start),
                    "end": float(turn.end)
                })
            
            num_speakers = len(set(s['speaker'] for s in segments))
            logging.info(f"Found {num_speakers} speakers and {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logging.error(f"Diarization failed: {e}")
            raise

    def diarize_and_split(self, audio_path, min_speakers=None, max_speakers=None, 
                           output_dir=None, export_json=True):
        """Diarize audio and split it into speaker segments
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            output_dir: Directory to save the split audio segments (optional)
            export_json: Whether to export diarization results to JSON (default: True)
            
        Returns:
            Tuple of (segments, segment_files) where segments is a list of diarization
            segments and segment_files is a dictionary mapping segment indices to file paths
        """
        # Perform diarization
        segments = self.diarize(audio_path, min_speakers, max_speakers)
        
        # Create output directory for segments if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            logging.info(f"Created temporary output directory: {output_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Splitting audio into segments in: {output_dir}")
        
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Split the audio into segments
        segment_files = {}
        for i, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            speaker = segment["speaker"]
            
            # Extract segment
            audio_segment = audio[start_ms:end_ms]
            
            # Create segment file path
            filename = f"{i:04d}_{speaker}_{start_ms}_{end_ms}.wav"
            segment_path = os.path.join(output_dir, filename)
            
            # Export segment
            audio_segment.export(segment_path, format="wav")
            
            # Store segment file path
            segment_files[i] = segment_path
            
            # Add file path to segment data
            segments[i]["file"] = segment_path
        
        # Export diarization results to JSON if requested
        if export_json:
            json_path = os.path.join(output_dir, "diarization.json")
            with open(json_path, 'w') as f:
                json.dump(segments, f, indent=2)
            logging.info(f"Exported diarization data to: {json_path}")
        
        return segments, segment_files

    def merge_transcripts_with_diarization(self, segments, transcripts):
        """Merge transcription results with diarization segments
        
        Args:
            segments: List of diarization segments
            transcripts: Dictionary mapping segment indices to transcription text
            
        Returns:
            Merged transcript with speaker labels
        """
        # Ensure segments are sorted by start time
        sorted_segments = sorted(enumerate(segments), key=lambda x: x[1]["start"])
        
        # Merge transcripts
        result = []
        current_speaker = None
        current_text = []
        
        for i, segment in sorted_segments:
            if i not in transcripts:
                continue
                
            speaker = segment["speaker"]
            text = transcripts[i].strip()
            
            if not text:
                continue
                
            if speaker != current_speaker:
                # New speaker
                if current_speaker is not None and current_text:
                    result.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = speaker
                current_text = [text]
            else:
                # Same speaker
                current_text.append(text)
        
        # Add the last speaker's text
        if current_speaker is not None and current_text:
            result.append(f"{current_speaker}: {' '.join(current_text)}")
        
        return "\n\n".join(result)

    def export_to_formats(self, segments, transcripts, output_path_base, formats=None):
        """Export diarized transcription to various formats
        
        Args:
            segments: List of diarization segments
            transcripts: Dictionary mapping segment indices to transcription text
            output_path_base: Base path for output files (without extension)
            formats: List of formats to export (default: ["txt", "srt", "vtt"])
            
        Returns:
            Dictionary mapping formats to output file paths
        """
        if formats is None:
            formats = ["txt", "srt", "vtt"]
            
        # Ensure segments are sorted by start time
        sorted_segments = sorted(enumerate(segments), key=lambda x: x[1]["start"])
        
        # Create mapping of exported files
        exported_files = {}
        
        # Process each requested format
        for fmt in formats:
            if fmt.lower() == "txt":
                # Plain text format
                txt_path = f"{output_path_base}.txt"
                merged_transcript = self.merge_transcripts_with_diarization(segments, transcripts)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(merged_transcript)
                exported_files["txt"] = txt_path
                
            elif fmt.lower() == "srt":
                # SRT format
                srt_path = f"{output_path_base}.srt"
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for i, (idx, segment) in enumerate(sorted_segments, 1):
                        if idx not in transcripts:
                            continue
                            
                        text = transcripts[idx].strip()
                        if not text:
                            continue
                            
                        start = segment["start"]
                        end = segment["end"]
                        speaker = segment["speaker"]
                        
                        # Format timestamps as HH:MM:SS,mmm
                        start_time = time_to_srt(start)
                        end_time = time_to_srt(end)
                        
                        # Write SRT entry
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{speaker}: {text}\n\n")
                
                exported_files["srt"] = srt_path
                
            elif fmt.lower() == "vtt":
                # VTT format (WebVTT)
                vtt_path = f"{output_path_base}.vtt"
                with open(vtt_path, 'w', encoding='utf-8') as f:
                    # Write VTT header
                    f.write("WEBVTT\n\n")
                    
                    for i, (idx, segment) in enumerate(sorted_segments, 1):
                        if idx not in transcripts:
                            continue
                            
                        text = transcripts[idx].strip()
                        if not text:
                            continue
                            
                        start = segment["start"]
                        end = segment["end"]
                        speaker = segment["speaker"]
                        
                        # Format timestamps as HH:MM:SS.mmm
                        start_time = time_to_vtt(start)
                        end_time = time_to_vtt(end)
                        
                        # Write VTT entry
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{speaker}: {text}\n\n")
                
                exported_files["vtt"] = vtt_path
                
            elif fmt.lower() == "json":
                # JSON format
                json_path = f"{output_path_base}.json"
                output_data = []
                
                for idx, segment in sorted_segments:
                    if idx not in transcripts:
                        continue
                        
                    text = transcripts[idx].strip()
                    if not text:
                        continue
                        
                    output_data.append({
                        "speaker": segment["speaker"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": text
                    })
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
                
                exported_files["json"] = json_path
        
        return exported_files

def process_with_progress(audio_file, pipeline, num_speakers=None):
    """Process audio with enhanced progress tracking"""
    # Initialize progress tracker
    progress = EnhancedProgress(audio_file)
    
    try:
        # Set up diarization parameters
        diarization_kwargs = {}
        if num_speakers is not None:
            diarization_kwargs['num_speakers'] = num_speakers
            print(f"\nUsing fixed number of speakers: {num_speakers}")
        
        # Try to use PyAnnote's progress hook if available
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            
            with ProgressHook() as hook:
                # Define progress callback
                def update_progress(step):
                    stage_name = step.name if hasattr(step, 'name') else "Processing"
                    percentage = step.percentage if hasattr(step, 'percentage') else None
                    progress.update_stage(stage_name, percentage)
                
                # Register callback
                hook.on_step = update_progress
                
                # Add hook to kwargs
                diarization_kwargs['hook'] = hook
                
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
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    pipeline = pipeline.to(device)
    
    # Apply optimizations based on device
    if str(device) == 'mps':
        try:
            # Optimize segmentation parameters for faster processing
            if hasattr(pipeline, 'segmentation'):
                pipeline.segmentation.duration = 5.0  # Longer window for better results
                # Try to use larger step size for faster processing (if available)
                if hasattr(pipeline.segmentation, 'step'):
                    pipeline.segmentation.step = 0.5  # Default is usually 0.1s, 5x faster
            
            # Optimize embedding batch size
            if hasattr(pipeline, 'embedding'):
                # Optimal batch size for M1/M2
                pipeline.embedding.batch_size = 32
            
            # Optimize clustering
            if hasattr(pipeline, 'clustering'):
                # Use faster clustering method
                if hasattr(pipeline.clustering, 'method'):
                    pipeline.clustering.method = 'centroid'
            
            print("✅ Applied MPS-specific optimizations")
        except Exception as e:
            print(f"⚠️ Could not apply all optimizations: {e}")
    elif str(device) == 'cuda':
        try:
            # CUDA-specific optimizations
            if hasattr(pipeline, 'embedding'):
                pipeline.embedding.batch_size = 64  # Larger batch for CUDA
            
            print("✅ Applied CUDA-specific optimizations")
        except Exception:
            pass
    else:
        # CPU-specific optimizations
        if hasattr(pipeline, 'embedding'):
            pipeline.embedding.batch_size = 16  # Smaller batch for CPU
    
    return pipeline

# Helper functions
def time_to_srt(seconds):
    """Convert time in seconds to SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def time_to_vtt(seconds):
    """Convert time in seconds to VTT format (HH:MM:SS.mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def verify_authentication(token=None):
    """Test connection to Hugging Face - directly using the proven test script approach"""
    if not token:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        try:
            token_path = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    token = f.read().strip()
        except:
            pass
    
    if not token:
        print("❌ No token found. Please provide a token via argument, environment variable, or ~/.huggingface/token file.")
        return False
    
    try:
        # We DON'T call login() here directly - not needed for testing
        print(f"Token length: {len(token)} characters")
    except Exception as e:
        print(f"❌ Token error: {e}")
        return False
    
    api = HfApi()
    models_to_check = {
        "pyannote/segmentation": "config.yaml",
        "pyannote/speaker-diarization": "config.yaml",
        "pyannote/speaker-diarization-3.1": "config.yaml",
    }
    
    all_access = True
    
    for model, test_file in models_to_check.items():
        try:
            info = api.model_info(model, token=token)
            print(f"✅ Can access model info for {model}")
            
            # Check if the file exists in the list of siblings
            available_files = [s.rfilename for s in info.siblings]
            if test_file not in available_files:
                print(f"  ⚠️ File '{test_file}' not found in model '{model}' -- available files: {available_files[:5]}...")
                
                # Try to find a config file
                config_files = [f for f in available_files if f.startswith("config")]
                if config_files:
                    test_file = config_files[0]
                    print(f"  ℹ️ Will try alternative file: {test_file}")
                else:
                    all_access = False
                    continue
            
            # Try downloading the file
            try:
                hf_hub_download(model, test_file, token=token, local_files_only=False)
                print(f"  ✅ Successfully downloaded '{test_file}' from {model}")
            except Exception as e:
                print(f"  ❌ Failed to download '{test_file}' from {model}: {e}")
                all_access = False

        except Exception as e:
            print(f"❌ Cannot access model {model}: {e}")
            all_access = False
            if "403" in str(e) or "401" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"  ⚠️ This model is gated. Visit and request access:")
                print(f"  🔗 https://huggingface.co/{model}")
    
    if all_access:
        print("\n✅ All models can be accessed! Your authentication is working correctly.")
        return True
    else:
        print("\n⚠️ Some models could not be accessed properly.")
        print("Please ensure you've accepted any required license agreements or requested access:")
        for model in models_to_check:
            print(f"- https://huggingface.co/{model}")
        return False

def test_diarization(audio_path, hf_token=None, device=None, num_speakers=None):
    """Test the diarization functionality"""
    logging.info(f"Starting speaker diarization...")
    manager = DiarizationManager(hf_token=hf_token, device=device)
    
    # First verify authentication directly
    token = manager._get_token()
    if token:
        logging.info(f"Using token with length: {len(token)} characters")
    else:
        logging.warning("No token found!")
    
    try:
        # If num_speakers is provided, use it directly
        if num_speakers is not None:
            segments = manager.diarize(audio_path, max_speakers=num_speakers)
            segments, segment_files = manager.diarize_and_split(audio_path, max_speakers=num_speakers)
        else:
            segments, segment_files = manager.diarize_and_split(audio_path)
        
        logging.info(f"Diarization completed successfully on {audio_path}")
        logging.info(f"Found {len(set(s['speaker'] for s in segments))} speakers in {len(segments)} segments")
        
        # Print the first few segments
        for i, segment in enumerate(segments[:5]):
            logging.info(f"{i}: {segment['speaker']} ({segment['start']:.2f}s - {segment['end']:.2f}s)")
        
        return segments, segment_files
    except Exception as e:
        logging.error(f"Speaker diarization process failed: {e}")
        return None, None

if __name__ == "__main__":
    print ("Audio diarization utility")
    if len(sys.argv) < 2:
        print("Usage: python diarize_audio.py <audio_file> [huggingface_token] [device] [num_speakers]")
        print("       python diarize_audio.py --verify-auth [huggingface_token]")
        sys.exit(1)
    
    if sys.argv[1] == "--verify-auth":
        token = sys.argv[2] if len(sys.argv) > 2 else None
        verify_authentication(token)
        sys.exit(0)
    
    audio_path = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None
    device = sys.argv[3] if len(sys.argv) > 3 else None
    num_speakers = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else None

    print(f"Parameters: audio {audio_path}, token: {'provided' if hf_token else 'not provided'}, device: {device}, num_speakers: {num_speakers}")
    
    test_diarization(audio_path, hf_token, device, num_speakers)
