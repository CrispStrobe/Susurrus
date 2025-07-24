# Susurrus: Whisper Audio Transcription GUI

Susurrus is a flexible audio transcription frontend that leverages various AI models, mostly based on OpenAI Whisper, and backends to convert speech to text. It transcribes audio files, including online content, using a number of optional models and pipelines.

## Features

- Support for multiple transcription backends (mlx-whisper, OpenAI Whisper, faster-whisper, transformers, whisper.cpp, ctranslate2, whisper-jax, insanely-fast-whisper)
- **Speaker diarization** with language-specific models (English, German, and others)
- Audio file upload and URL input support
- YouTube audio extraction and transcription
- Proxy support for network requests
- Language selection for targeted transcription
- Transcription metrics and progress tracking
- Graphical user interface
- Advanced options including start/end time for transcription, max chunk length, and output format selection for whisper.cpp (enabling subtitle export)
- Audio trimming functionality

## Screenshot

![Susurrus Interface](susurrus.png)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- C++ compiler (for whisper.cpp)
- CMake (for whisper.cpp)
- FFmpeg
- **Windows only**: Chocolatey package manager (recommended)
- **For whisper-jax**: Rust toolchain
- **For speaker diarization**: Hugging Face account and API token

### Windows Installation

1. **Install Chocolatey** (if not already installed):
   - Open PowerShell as Administrator
   - Run: `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))`

2. **Install required tools**:
   ```powershell
   choco install cmake
   choco install rust      # Required for whisper-jax
   choco install mingw     # Required for tokenizers compilation
   choco install ffmpeg
   ```

3. **Restart PowerShell** to refresh environment variables

4. **Clone the repository**:
   ```powershell
   git clone https://github.com/CrispStrobe/susurrus.git
   cd susurrus
   ```

5. **Create and activate virtual environment**:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

6. **Install base requirements**:
   ```powershell
   pip install -r requirements.txt
   ```

7. **Install whisper-jax from source**:
   ```powershell
   pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
   ```

8. **Build whisper.cpp**:
   ```powershell
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   mkdir build
   cd build
   cmake -B . -DCMAKE_CXX_FLAGS="/utf-8" -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   cd ../..
   ```

9. **Set up Hugging Face token for speaker diarization** (optional):
    - Create a free account at [https://huggingface.co](https://huggingface.co)
    - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    - Create a new token with 'read' access
    - Accept the user agreement at [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
    - Set the token as an environment variable:
      ```powershell
      setx HF_TOKEN "your_token_here"
      ```
    - Or enter it directly in the application interface

### macOS/Linux Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CrispStrobe/susurrus.git
   cd susurrus
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional backend-specific packages**:
   ```bash
   pip install openai-whisper faster-whisper transformers ctranslate2 whisper-jax soundfile insanely-fast-whisper
   ```

5. **Install whisper.cpp**:
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   cd ../..
   ```

6. **Install FFmpeg**:
   - **macOS**: `brew install ffmpeg`
   - **Linux (Ubuntu/Debian)**: `sudo apt-get update && sudo apt-get install ffmpeg`

7. **Set up Hugging Face token for speaker diarization** (optional):
   - Create a free account at [https://huggingface.co](https://huggingface.co)
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with 'read' access
   - Accept the user agreement at [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - Set the token as an environment variable:
     ```bash
     export HF_TOKEN="your_token_here"
     # Add to ~/.bashrc or ~/.zshrc to make it permanent
     echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
     ```

### Manual Windows Installation (Alternative)

If you prefer not to use Chocolatey:

1. **Install tools manually**:
   - **CMake**: Download from [https://cmake.org/download/](https://cmake.org/download/) and add to PATH
   - **Rust**: Download from [https://rustup.rs/](https://rustup.rs/)
   - **Visual Studio**: Install Visual Studio with C++ support, or install MinGW-w64
   - **FFmpeg**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add bin folder to PATH

2. Follow steps 4-9 from the Windows Installation section above

## Usage

1. **Activate the virtual environment** (if not already activated):
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

2. **Run the main application**:
   ```bash
   python susurrus.py
   ```

3. **Use the graphical interface to**:
   - Upload an audio file or provide a URL
   - Select the desired transcription backend and model
   - **Enable speaker diarization** (optional):
     - Check "Enable Speaker Diarization"
     - Enter your Hugging Face token (if not set in environment)
     - Choose language-specific diarization model if needed
     - Set min/max speakers if known
   - Configure advanced options if needed
   - Start the transcription process

4. **View the transcription results** and metrics in the application window

5. **Save the transcription** to a text file using the "Save" button

### Running the Transcription Worker Script

The transcription worker script can be run separately for debugging or advanced usage:

```bash
python transcribe_worker.py --audio-input <audio_file> --audio-url <url> --model-id <model_id> --word-timestamps --language <lang> --backend <backend> --device <device> --pipeline-type <type> --max-chunk-length <length> --output-format <format> --quantization <quant_type> --batch-size <size> --preprocessor-path <path> --original-model-id <orig_id> --start-time <start> --end-time <end>
```

**Example**:
```bash
python transcribe_worker.py --audio-input input.wav --model-id mlx-community/whisper-large-v3-mlx --word-timestamps --language en --backend mlx-whisper --device auto --pipeline-type default --start-time 10 --end-time 60
```

### Speaker Diarization

Susurrus supports speaker diarization to identify different speakers in audio recordings:

**Setup Requirements**:
1. Install required dependencies: `pip install pyannote.audio`
2. Get a Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the model license at [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)

**Usage Tips**:
- Choose language-specific models for better results
- Set min/max speakers if you know the expected number

## Troubleshooting

### Windows-Specific Issues

**whisper-jax installation fails**:
- Ensure Rust is installed: `cargo --version`
- Install MinGW for GCC: `choco install mingw`
- Try installing pre-built tokenizers first: `pip install "tokenizers>=0.14.0,<0.15.0"`

**Git ownership warnings**:
```powershell
git config --global --add safe.directory "C:/path/to/your/susurrus/directory"
```

**Hugging Face token issues**:
- Ensure you have a valid token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Accept the model license at [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
- Set environment variable: `setx HF_TOKEN "your_token_here"`
- Restart the application after setting the token

**refreshenv not working in PowerShell**:
```powershell
Import-Module $env:ChocolateyInstall\helpers\chocolateyProfile.psm1
refreshenv
```

### Common Issues

**FFmpeg not found**:
- Ensure FFmpeg is installed and in your system PATH
- Test with: `ffmpeg -version`

**CUDA/GPU issues**:
- Ensure you have compatible NVIDIA drivers
- Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

**Memory issues**:
- Try using smaller models
- Reduce batch size in advanced options
- Use CPU instead of GPU for very large files

**Speaker diarization issues**:
- Ensure pyannote.audio is installed: `pip install pyannote.audio`
- Check that your HF token has 'read' access
- Verify you accepted the license agreement
- First run downloads ~1GB model, ensure good internet connection

## System Requirements

- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for models and dependencies  
- **GPU**: NVIDIA GPU with CUDA support recommended for faster processing
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [MLX Community](https://github.com/ml-explore/mlx-examples)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Transformers](https://github.com/huggingface/transformers)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
