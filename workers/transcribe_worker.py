#!/usr/bin/env python3
# transcribe_worker.py - Worker script for audio transcription
#!/usr/bin/env python3
"""Standalone transcription worker script - Refactored"""
import sys
import os
import logging
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.transcription.utils import TranscriptionUtils
from workers.transcription.backends import get_backend
from utils.audio_utils import convert_audio_to_wav, trim_audio, is_valid_time
from utils.download_utils import download_audio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

def main():
    """Main transcription worker"""
    # Track temporary files
    temp_files = []
    
    try:
        # Parse arguments
        args = TranscriptionUtils.parse_arguments()
        
        # Validate arguments
        if not args.audio_input and not args.audio_url:
            logging.error("Either --audio-input or --audio-url must be provided")
            sys.exit(1)
        
        if args.audio_input and not os.path.exists(args.audio_input):
            logging.error(f"Audio file not found: {args.audio_input}")
            sys.exit(1)
        
        # Clean up language parameter
        if args.language is not None and args.language.strip() == '':
            args.language = None
        
        # Determine device
        device = TranscriptionUtils.detect_device(args.device)
        logging.info(f"Starting transcription on device: {device} using backend: {args.backend}")
        
        # Set up proxies if needed
        proxies = TranscriptionUtils.setup_proxies(
            args.proxy_url, args.proxy_username, args.proxy_password
        )
        
        # Handle audio input
        if args.audio_input and os.path.exists(args.audio_input):
            original_audio_path = args.audio_input
        elif args.audio_url:
            logging.info(f"Downloading audio from URL: {args.audio_url}")
            original_audio_path = download_audio(args.audio_url, proxies=proxies, 
                                                ffmpeg_path=args.ffmpeg_path)
            if not original_audio_path:
                raise Exception(f"Failed to download audio from {args.audio_url}")
            temp_files.append(original_audio_path)
        else:
            raise Exception("No valid audio source provided.")
        
        # Trim audio if requested
        if is_valid_time(args.start_time) or is_valid_time(args.end_time):
            logging.info(f"Trimming audio from {args.start_time}s to {args.end_time}s")
            working_audio_path = trim_audio(original_audio_path, args.start_time, args.end_time)
            if working_audio_path != original_audio_path:
                temp_files.append(working_audio_path)
        else:
            working_audio_path = original_audio_path
        
        # Set default model if not provided
        if args.model_id is None:
            from config import get_default_model_for_backend
            args.model_id = get_default_model_for_backend(args.backend)
            logging.info(f"Using default model: {args.model_id}")
        
        logging.info(f"Model: {args.model_id}")
        if args.language:
            logging.info(f"Language: {args.language}")
        
        # Get backend instance
        start_time = time.time()
        
        backend = get_backend(
            args.backend,
            model_id=args.model_id,
            device=device,
            language=args.language,
            word_timestamps=args.word_timestamps,
            max_chunk_length=args.max_chunk_length,
            output_format=args.output_format,
            quantization=args.quantization,
            preprocessor_path=args.preprocessor_path,
            original_model_id=args.original_model_id,
            mistral_api_key=args.mistral_api_key,
            temperature=args.temperature,
        )
        
        # Preprocess audio if backend requires it
        audio_to_transcribe = backend.preprocess_audio(working_audio_path)
        if audio_to_transcribe != working_audio_path:
            temp_files.append(audio_to_transcribe)
        
        # Transcribe
        for result in backend.transcribe(audio_to_transcribe):
            if len(result) == 3:
                start, end, text = result
                print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
            else:
                # Some backends might yield just text
                print(result, flush=True)
        
        # Cleanup
        backend.cleanup()
        
        # Print stats
        transcription_time = time.time() - start_time
        print(f"Total transcription time: {transcription_time:.2f} seconds", flush=True)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")

if __name__ == "__main__":
    main()