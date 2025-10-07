@echo off
REM install_voxtral.bat - Install Voxtral dependencies (Windows)

echo Installing Voxtral dependencies...

REM Uninstall existing transformers
echo Removing existing transformers installation...
pip uninstall transformers -y

REM Install transformers from git
echo Installing latest transformers from GitHub...
pip install git+https://github.com/huggingface/transformers.git

REM Install audio dependencies
echo Installing audio processing dependencies...
pip install mistral-common[audio]
pip install librosa soundfile

REM Verify installation
echo.
echo Verifying installation...
python -c "try: from transformers import VoxtralForConditionalGeneration, AutoProcessor; print('✅ Voxtral is available!'); print('✅ Installation successful!'); except ImportError as e: print('❌ Installation failed:', e); exit(1)"

echo.
echo Installation complete!
echo.
echo To use Voxtral locally:
echo   python transcribe_worker.py --backend voxtral-local --audio-input file.mp3
echo.
echo To use Voxtral API:
echo   set MISTRAL_API_KEY=your-key-here
echo   python transcribe_worker.py --backend voxtral-api --audio-input file.mp3

pause