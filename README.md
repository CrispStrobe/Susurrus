# Susurrus: Audio Transcription Suite

Susurrus is a professional, modular audio transcription application that leverages various AI models and backends to convert speech to text. Built with a clean architecture, it supports multiple Whisper implementations, speaker diarization, and extensive customization options.

## ✨ Features

### Core Transcription
- **Multiple Backend Support**: mlx-whisper, OpenAI Whisper, faster-whisper, transformers, whisper.cpp, ctranslate2, whisper-jax, insanely-fast-whisper, Voxtral
- **Flexible Input**: Local files, URLs, also of videos
- **Audio Format Support**: MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, WebM, MP4, WMA
- **Language Detection**: Automatic or manual language selection
- **Time-based Trimming**: Transcribe specific portions of audio
- **Word-level Timestamps**: Precise timing information (backend-dependent)

### Speaker Diarization
- **Multi-speaker Identification**: Automatically detect and label different speakers
- **Language-specific Models**: Optimized models for English, German, Chinese, Spanish, Japanese
- **Configurable Parameters**: Set min/max speaker counts
- **Multiple Output Formats**: TXT, SRT, VTT, JSON with speaker labels
- **PyAnnote.audio Integration**: State-of-the-art diarization engine

### Voxtral Support (New!)
- **Voxtral Local**: On-device inference with Mistral's speech model
- **Voxtral API**: Cloud-based inference via Mistral AI API
- **8 Language Support**: EN, FR, ES, DE, IT, PT, PL, NL
- **Long Audio Processing**: Automatic chunking for files over 25 minutes

### Advanced Features
- **Proxy Support**: HTTP/SOCKS5 proxy for network requests
- **Device Selection**: Auto-detect or manually choose CPU/GPU/MPS
- **Model Conversion**: Automatic CTranslate2 model conversion
- **Progress Tracking**: Real-time progress with ETA estimation
- **Settings Persistence**: Save your preferences between sessions
- **Dependency Management**: Built-in installer for missing components
- **CUDA Diagnostics**: Detailed GPU/CUDA troubleshooting tools

## 📦 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/CrispStrobe/Susurrus.git
cd Susurrus

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
# Or as a module:
python -m susurrus
```

### Prerequisites

- **Python 3.8+**
- **FFmpeg** (for audio format conversion)
- **Git**
- **C++ compiler** (for whisper.cpp, optional)
- **CUDA Toolkit** (for GPU acceleration, optional)

### Platform-Specific Setup

#### Windows
```powershell
# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install dependencies
choco install cmake ffmpeg git python

# For GPU support
choco install cuda
```

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ffmpeg cmake python git

# For Apple Silicon optimization
pip install mlx mlx-whisper
```

#### Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt update
sudo apt install ffmpeg cmake build-essential python3 python3-pip git

# For GPU support
# Follow CUDA installation guide for your distribution
```

### Optional Backend Installation

```bash
# MLX (Apple Silicon only)
pip install mlx-whisper

# Faster Whisper (recommended)
pip install faster-whisper

# Transformers
pip install transformers torch torchaudio

# Whisper.cpp (manual build required)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && mkdir build && cd build
cmake .. && make

# CTranslate2
pip install ctranslate2

# Whisper-JAX
pip install whisper-jax

# Insanely Fast Whisper
pip install insanely-fast-whisper

# Voxtral (requires dev transformers)
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers.git
pip install mistral-common[audio] soundfile
```

### Speaker Diarization Setup

```bash
# Install pyannote.audio
pip install pyannote.audio

# Get Hugging Face token
# 1. Sign up at https://huggingface.co
# 2. Create token at https://huggingface.co/settings/tokens
# 3. Accept license at https://huggingface.co/pyannote/speaker-diarization

# Set token (choose one method):
# Method 1: Environment variable
export HF_TOKEN="your_token_here"  # Linux/macOS
setx HF_TOKEN "your_token_here"    # Windows

# Method 2: Config file
mkdir -p ~/.huggingface
echo "your_token_here" > ~/.huggingface/token

# Method 3: Enter in GUI
```

### Voxtral API Setup

```bash
# Get Mistral API key from https://console.mistral.ai/

# Set API key (choose one method):
# Method 1: Environment variable
export MISTRAL_API_KEY="your_key_here"  # Linux/macOS
setx MISTRAL_API_KEY "your_key_here"    # Windows

# Method 2: Config file
mkdir -p ~/.mistral
echo "your_key_here" > ~/.mistral/api_key

# Method 3: Enter in GUI
```

## 🚀 Usage

### GUI Application

```bash
# Start the application
python main.py

# Or as a module
python -m susurrus
```

**Basic Workflow**:
1. **Select Audio Source**: Choose file or enter URL
2. **Choose Backend**: Select transcription engine
3. **Configure Options**: Set language, model, device
4. **Enable Diarization** (optional): Identify speakers
5. **Start Transcription**: Click "Transcribe"
6. **Save Results**: Export to TXT, SRT, or VTT

### Command Line Workers

#### Transcription Worker
```bash
python workers/transcribe_worker.py \
  --audio-input audio.mp3 \
  --backend faster-batched \
  --model-id large-v3 \
  --language en \
  --device auto
```

#### Diarization Worker
```bash
python workers/diarize_worker.py \
  --audio-input audio.mp3 \
  --hf-token YOUR_TOKEN \
  --transcribe \
  --model-id base \
  --backend faster-batched \
  --output-formats txt,srt,vtt
```

### Python API

```python
# Transcription backend example
from workers.transcription.backends import get_backend

backend = get_backend(
    'faster-batched',
    model_id='large-v3',
    device='auto',
    language='en'
)

for start, end, text in backend.transcribe('audio.mp3'):
    print(f"[{start:.2f}s -> {end:.2f}s] {text}")
```

```python
# Diarization example
from backends.diarization import DiarizationManager

manager = DiarizationManager(hf_token="YOUR_TOKEN")
segments, files = manager.diarize_and_split('audio.mp3')

for segment in segments:
    print(f"{segment['speaker']}: {segment['text']}")
```

## 🧪 Development

### Architecture Overview

```
susurrus/
├── main.py                    # Application entry point
├── config.py                  # Central configuration
├── backends/                  # Transcription & diarization backends
│   ├── diarization/          # Speaker diarization module
│   │   ├── manager.py        # Diarization orchestration
│   │   └── progress.py       # Enhanced progress tracking
│   └── transcription/        # Transcription backends
│       ├── voxtral_local.py  # Voxtral local inference
│       └── voxtral_api.py    # Voxtral API integration
├── gui/                       # User interface components
│   ├── main_window.py        # Main application window
│   ├── widgets/              # Custom widgets
│   │   ├── collapsible_box.py
│   │   ├── diarization_settings.py
│   │   ├── voxtral_settings.py
│   │   └── advanced_options.py
│   └── dialogs/              # Dialog windows
│       ├── dependencies_dialog.py
│       ├── installer_dialog.py
│       └── cuda_diagnostics_dialog.py
├── workers/                   # Background processing
│   ├── transcription_thread.py    # GUI thread wrapper
│   ├── transcribe_worker.py       # Standalone transcription worker
│   ├── diarize_worker.py          # Standalone diarization worker
│   └── transcription/             # Transcription backend implementations
│       ├── backends/
│       │   ├── base.py           # Base backend interface
│       │   ├── mlx_backend.py
│       │   ├── faster_whisper_backend.py
│       │   ├── transformers_backend.py
│       │   ├── whisper_cpp_backend.py
│       │   ├── ctranslate2_backend.py
│       │   ├── whisper_jax_backend.py
│       │   ├── insanely_fast_backend.py
│       │   ├── openai_whisper_backend.py
│       │   └── voxtral_backend.py
│       └── utils.py
├── utils/                     # Utility modules
│   ├── device_detection.py   # CUDA/MPS/CPU detection
│   ├── audio_utils.py        # Audio processing utilities
│   ├── download_utils.py     # URL downloading
│   ├── dependency_check.py   # Dependency verification
│   └── format_utils.py       # Time formatting utilities
├── models/                    # Model configuration
│   └── model_config.py       # Model mappings & utilities
└── scripts/                   # Standalone utility scripts
    ├── test_voxtral.py       # Voxtral testing
    └── pyannote_torch26.py   # PyTorch 2.6+ compatibility
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_backends.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .
pylint susurrus/

# Type checking
mypy .
```

### Adding a New Backend

1. Create a new file in `workers/transcription/backends/`
2. Inherit from `TranscriptionBackend`
3. Implement required methods:
   ```python
   class MyBackend(TranscriptionBackend):
       def transcribe(self, audio_path):
           # Yield (start, end, text) tuples
           pass
       
       def preprocess_audio(self, audio_path):
           # Optional preprocessing
           return audio_path
       
       def cleanup(self):
           # Optional cleanup
           pass
   ```
4. Register in `workers/transcription/backends/__init__.py`
5. Add to `config.py` BACKEND_MODEL_MAP

## 🔧 Configuration

### Settings Location

- **Windows**: `%APPDATA%\Susurrus\AudioTranscription.ini`
- **macOS**: `~/Library/Preferences/com.Susurrus.AudioTranscription.plist`
- **Linux**: `~/.config/Susurrus/AudioTranscription.conf`

### Environment Variables

- `HF_TOKEN`: Hugging Face API token (diarization)
- `MISTRAL_API_KEY`: Mistral AI API key (Voxtral)
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO`: MPS memory optimization
- `CUDA_VISIBLE_DEVICES`: GPU selection

## 📊 Performance Tips

### GPU Acceleration

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon Optimization

```bash
# Use MLX backend for best performance
pip install mlx-whisper

# Or use MPS device with other backends
# Will auto-detect in GUI
```

### Memory Management

- Use smaller models for limited RAM
- Enable chunking for long audio files
- Use `faster-batched` backend with appropriate batch size
- Close other applications during processing

## 🐛 Troubleshooting

### Common Issues

**"No module named 'X'"**
```bash
pip install X
```

**FFmpeg not found**
```bash
# Verify installation
ffmpeg -version

# Add to PATH if needed (Windows)
setx PATH "%PATH%;C:\path\to\ffmpeg\bin"
```

**CUDA errors**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use Tools > CUDA Diagnostics in GUI for detailed info
```

**Diarization authentication fails**
```bash
# Verify token
python -c "from huggingface_hub import HfApi; HfApi().whoami(token='YOUR_TOKEN')"

# Accept license
# Visit: https://huggingface.co/pyannote/speaker-diarization
```

**PyTorch 2.6+ compatibility issues**
```bash
# Run the compatibility script
python scripts/pyannote_torch26.py
```

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Susurrus.git
cd Susurrus

# Create feature branch
git checkout -b feature-name

# Install dev dependencies
pip install -r requirements-dev.txt

# Make changes and test
pytest tests/

# Submit PR
```

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon acceleration
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized inference
- [PyAnnote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Mistral AI](https://mistral.ai/) - Voxtral model
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - For URL downloading
