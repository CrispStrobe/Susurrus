# transcribe_worker.py
import argparse
import sys
import os
import logging
import torch
from pydub import AudioSegment
import subprocess
import requests
import tempfile
import time
import yt_dlp
from urllib.parse import urlparse

from pydub.utils import mediainfo

def download_audio(url):
    parsed_url = urlparse(url)
    logging.info(f"Downloading audio from URL: {url}")
    audio_file = None

    download_methods = [download_with_yt_dlp, download_with_pytube, download_with_ffmpeg]

    for method in download_methods:
        try:
            audio_file = method(url)
            if audio_file and os.path.exists(audio_file):
                logging.info(f"Audio downloaded successfully using {method.__name__}")
                return audio_file, True  # Return the path and a flag indicating it's a temporary file
        except Exception as e:
            logging.error(f"{method.__name__} failed: {str(e)}")
            continue  # Try the next method

    logging.error(f"All download methods failed for URL: {url}")
    return None, False

def download_with_yt_dlp(url):
    logging.info("Trying to download using yt-dlp...")
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, '%(id)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if 'entries' in info:
            # Can be a playlist or a list of videos
            info = info['entries'][0]
        output_file = ydl.prepare_filename(info)
        output_file = os.path.splitext(output_file)[0] + '.mp3'
        if os.path.exists(output_file):
            logging.info(f"Downloaded YouTube audio: {output_file}")
            return output_file
        else:
            raise Exception("yt-dlp did not produce an output file.")

def download_with_pytube(url):
    logging.info("Trying to download using pytube...")
    from pytube import YouTube
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    if audio_stream is None:
        raise Exception("No audio streams available.")
    temp_dir = tempfile.mkdtemp()
    out_file = audio_stream.download(output_path=temp_dir)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    logging.info(f"Downloaded and converted audio to: {new_file}")
    return new_file

def download_with_ffmpeg(url):
    logging.info("Trying to download using ffmpeg...")
    output_file = tempfile.mktemp(suffix='.mp3')
    command = ['ffmpeg', '-i', url, '-q:a', '0', '-map', 'a', output_file]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(output_file):
        logging.info(f"Downloaded audio using ffmpeg: {output_file}")
        return output_file
    else:
        raise Exception("ffmpeg did not produce an output file.")

def detect_audio_format(audio_path):
    info = mediainfo(audio_path)
    return info.get('format_name', 'wav')

def convert_to_ctranslate2_model(model_id, model_path, quantization):
    import ctranslate2
    import transformers

    logging.info(f"Converting model '{model_id}' to CTranslate2 format at '{model_path}' with quantization '{quantization}'")
    try:
        converter = ctranslate2.converters.TransformersConverter.from_model(
            model_name_or_path=model_id,
            load_as_float16=quantization in ['int8_float16', 'int8_float32']
        )
        converter.convert(model_path, quantization=quantization)
        logging.info(f"Model converted and saved to '{model_path}'")
    except Exception as e:
        logging.error(f"Failed to convert model: {str(e)}")
        sys.exit(1)

def convert_audio_to_wav(audio_path):
    try:
        # Detect audio format
        audio_format = detect_audio_format(audio_path)
        sound = AudioSegment.from_file(audio_path, format=audio_format)
        sound = sound.set_channels(1)  # Ensure mono audio
        sound = sound.set_sample_width(2)  # 2 bytes = 16 bits
        sound = sound.set_frame_rate(16000)  # Set sample rate to 16kHz
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        sound.export(temp_wav_file, format='wav')
        return temp_wav_file
    except Exception as e:
        logging.error(f"Error converting audio to 16-bit 16kHz WAV: {str(e)}")
        sys.exit(1)

def download_whisper_cpp_model_directly(model_file, model_path):
    base_url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
    url = base_url + model_file
    logging.info(f'Downloading {model_file} from {url}')
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f'Model downloaded to {model_path}')
    except Exception as e:
        logging.error(f"Failed to download model file: {str(e)}")
        sys.exit(1)

def find_whisper_cpp_executable():
    home_dir = os.path.expanduser('~')
    for root, dirs, files in os.walk(home_dir):
        if 'whisper.cpp' in dirs:
            whisper_cpp_dir = os.path.join(root, 'whisper.cpp')
            main_executable = os.path.join(whisper_cpp_dir, 'main')
            if os.path.isfile(main_executable) and os.access(main_executable, os.X_OK):
                return main_executable
    return None

def download_whisper_cpp_model(model_name, model_path):
    # Assume the script is located in ~/whisper.cpp/models/download-ggml-model.sh
    script_path = os.path.expanduser('~/whisper.cpp/models/download-ggml-model.sh')
    if not os.path.exists(script_path):
        logging.error(f"download-ggml-model.sh script not found at {script_path}")
        sys.exit(1)

    # Ensure the script is executable
    os.chmod(script_path, 0o755)
    
    # Run the script with the model name
    logging.info(f"Downloading model '{model_name}' using download-ggml-model.sh")
    try:
        subprocess.check_call([script_path, model_name], cwd=os.path.dirname(script_path))
        logging.info(f"Model '{model_name}' downloaded successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download model using download-ggml-model.sh: {str(e)}")
        sys.exit(1)
                              
def main():
    try:
        #print("transcription worker script started.")

        parser = argparse.ArgumentParser(description='Transcription worker script')
        parser.add_argument('--audio-input', help='Path to the audio input file')
        parser.add_argument('--audio-url', help='URL to the audio file')
        parser.add_argument('--model-id', required=True, help='Model ID to use')
        parser.add_argument('--word-timestamps', action='store_true', help='Enable word timestamps')
        parser.add_argument('--language', default=None, help='Language code')
        parser.add_argument('--backend', default='mlx-whisper', help='Backend to use')
        parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, gpu, mps)')
        parser.add_argument('--pipeline-type', default='default', help='Pipeline type')
        parser.add_argument('--max-chunk-length', type=float, default=0.0, help='Max chunk length in seconds')
        parser.add_argument('--output-format', default='txt', help='Output format for whisper.cpp')
        parser.add_argument('--quantization', default=None, help='Quantization type for ctranslate2 or faster-whisper (e.g., int8, float16)')
        parser.add_argument('--batch-size', type=int, default=16, help='Batch size for batched inference')

        args = parser.parse_args()

        if not args.audio_input and not args.audio_url:
            parser.error("Either --audio-input or --audio-url must be provided")

        audio_input = args.audio_input
        audio_url = args.audio_url
        model_id = args.model_id
        word_timestamps = args.word_timestamps
        language = args.language
        backend = args.backend
        device_arg = args.device
        pipeline_type = args.pipeline_type
        max_chunk_length = args.max_chunk_length
        output_format = args.output_format
        quantization = args.quantization

        # Handle empty or whitespace-only language input
        if language is not None and language.strip() == '':
            language = None    

        # Device selection
        if device_arg.lower() == 'auto':
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif device_arg.lower() == 'cpu':
            device = "cpu"
        elif device_arg.lower() == 'gpu':
            device = "cuda"
        elif device_arg.lower() == 'mps':
            device = "mps"
        else:
            device = "cpu"  # Default to CPU if unknown

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
        logging.info(f"Starting transcription on device: {device} using backend: {backend}")
        
        # Determine the audio source
        audio_path = None
        is_temp_file = False

        if audio_input and len(audio_input) > 0:
            # audio_input is a filepath to uploaded or recorded audio
            audio_path = audio_input
            is_temp_file = False
        elif audio_url and len(audio_url.strip()) > 0:
            # audio_url is provided
            audio_path, is_temp_file = download_audio(audio_url)
            if not audio_path:
                error_msg = f"Error downloading audio from {audio_url}. Check logs for details."
                logging.error(error_msg)
                sys.exit(1)
        else:
            error_msg = "No audio source provided. Please upload an audio file or enter a URL."
            logging.error(error_msg)
            sys.exit(1)

        # Now set audio_input to audio_path
        audio_input = audio_path
        
        start_time = time.time()

        if backend == 'mlx-whisper':
            # Existing mlx-whisper code
            import mlx_whisper
            transcribe_options = {
                "path_or_hf_repo": model_id,
                "verbose": True,  # Force verbose to be True to get incremental outputs
                "word_timestamps": word_timestamps,
                "language": language,
            }
            result = mlx_whisper.transcribe(audio_input, **transcribe_options)
        
        elif backend == 'faster-batched':
            
            from faster_whisper import WhisperModel, BatchedInferencePipeline

            # Handle device selection
            if device == 'mps':
                logging.warning("Faster-Whisper does not support MPS device. Switching to CPU.")
                device = 'cpu'

            # Determine compute_type based on quantization
            compute_type = args.quantization if args.quantization else 'int8'

            logging.info("Loading model...")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)
            pipeline = BatchedInferencePipeline(model=model)

            # Transcribe audio
            logging.info("Starting batched transcription")
            segments, info = pipeline.transcribe(audio_input, batch_size=args.batch_size)

            # Collect and print transcription
            for segment in segments:
                text = segment.text.strip()
                start = segment.start
                end = segment.end
                print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
        
        elif backend == 'faster-sequenced':
            
            if device == 'mps':
                logging.warning("Faster-Whisper does not support MPS device. Switching to CPU.")
                device = 'cpu'
            
            from faster_whisper import WhisperModel

            # Adjust compute_type based on the device
            if device == 'cuda':
                compute_type = "float16"
            elif device == 'cpu':
                compute_type = "int8"
            elif device == 'mps':
                compute_type = "int8"  # MPS may not support float16 in all cases
            else:
                compute_type = "int8"

            logging.info("Loading model...")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)

            options = {
                "language": language,
                "beam_size": 5,
                "best_of": 5,
                "word_timestamps": word_timestamps,
            }

            # 'faster-sequenced'
            segments, _ = model.transcribe(audio_input, **options)
            for segment in segments:
                # Obtain text and timestamps
                text = segment.text.strip()
                start = segment.start
                end = segment.end
                print(f'[{start:0>5.3f} --> {end:0>5.3f}] {text}', flush=True)
        
        elif backend == 'transformers':
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            if device == 'cpu':
                torch_dtype = torch.float32
            elif device == 'cuda':
                torch_dtype = torch.float16
            else:
                # For MPS, you might need to check compatibility
                torch_dtype = torch.float32
            
            logging.info("Loading model...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                return_timestamps="word" if word_timestamps else None,
                device=device,
            )
            logging.info("Starting transcription...")
            result = asr_pipeline(audio_input)
            if 'chunks' in result:
                for chunk in result['chunks']:
                    text = chunk['text'].strip()
                    start = chunk['timestamp'][0]
                    end = chunk['timestamp'][1]
                    print(f'[{start:0>5.3f} --> {end:0>5.3f}] {text}', flush=True)
            else:
                text = result['text'].strip()
                print(f'[00:00.000 --> XX:XX.XXX] {text}', flush=True)
        
        elif backend == 'whisper.cpp':
            
            # Map model_id to model file names
            model_file_map = {
                'large-v3-turbo-q5_0': 'ggml-large-v3-turbo-q5_0.bin',
                'large-v3-turbo': 'ggml-large-v3-turbo.bin',
                'small': 'ggml-small.bin',
                'base': 'ggml-base.bin',
                'tiny': 'ggml-tiny.bin',
                'tiny.en': 'ggml-tiny.en.bin',
                # Add other models as needed
            }

            model_file = model_file_map.get(model_id)
            if not model_file:
                logging.error(f"Model '{model_id}' is not recognized for whisper.cpp backend.")
                sys.exit(1)

            # Ensure model directory exists
            model_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_file)

            # Download the model if it's not already present
            if not os.path.exists(model_path):
                download_whisper_cpp_model(model_file, model_path)

            # Convert audio to 16kHz WAV
            logging.info("Converting audio to 16-bit 16kHz WAV format")
            audio_input = convert_audio_to_wav(audio_input)

            whisper_cpp_executable = find_whisper_cpp_executable()
            if not whisper_cpp_executable:
                logging.error("whisper.cpp executable not found in home directory or subfolders.")
                sys.exit(1)

            # Build the command
            cmd = [
                whisper_cpp_executable,
                '-m', model_path,
                '-f', audio_input,
                '--language', language if language else 'en',
                f'--output-{output_format}',
                #'--max-len', '1',  # For incremental output
                '--threads', str(os.cpu_count()),
                #'--quiet'  # Suppress progress logs
            ]
            
            
            logging.info(f"Running whisper.cpp with command: {' '.join(cmd)}")

            print (f"Running whisper.cpp with command: {' '.join(cmd)}")
            

            logging.info("Starting transcription with whisper.cpp")
            # Start the subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            # Read outputs from stdout as they are printed
            for line in process.stdout:
                text = line.strip()
                if text:
                    print(text, flush=True)

            # Wait for the process to complete and read stderr
            _, stderr_output = process.communicate()

            if process.returncode != 0:
                logging.error(stderr_output)
                sys.exit(1)
            else:
                # Only print OUTPUT FILE when output format is srt or vtt
                if output_format in ('srt', 'vtt'):
                    output_file = f"{audio_input}.{output_format}"
                    print(f"OUTPUT FILE: {output_file}", flush=True)
    
        elif backend == 'ctranslate2':
            import soundfile as sf
            import numpy as np
            from transformers import WhisperProcessor
            import ctranslate2

            quantization = args.quantization if args.quantization else 'int8_float16'

            logging.info("Loading model...")

            # Set model path
            model_dir = os.path.join(os.getcwd(), 'ctranslate2_models', model_id.replace('/', '_'))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                convert_to_ctranslate2_model(model_id, model_dir, quantization)

            translator = ctranslate2.models.Whisper(model_dir, device=device)
            processor = WhisperProcessor.from_pretrained(model_id)

            # Load and preprocess audio
            speech, sample_rate = sf.read(audio_input)
            input_features = processor(speech, sampling_rate=sample_rate, return_tensors="np").input_features[0]

            settings = {
                "beam_size": 5,
                "best_of": 5,
            }

            if language:
                settings["language"] = language

            logging.info("Starting transcription")

            result = translator.generate([input_features], **settings)
            transcription = processor.tokenizer.decode(result.sequences_ids[0])

            print(transcription, flush=True)
                
        elif backend == 'whisper-jax':
            try:
                import jax
                import jax.numpy as jnp
                from whisper_jax import FlaxWhisperPipline
                
                # Device selection
                if device.lower() == 'auto':
                    device = 'gpu' if jax.devices('gpu') else 'cpu'
                elif device.lower() in ['cpu', 'gpu']:
                    device = device.lower()
                else:
                    logging.warning("whisper-jax supports 'cpu' or 'gpu' devices. Using 'cpu'.")
                    device = 'cpu'

                # Set data type
                dtype = jnp.bfloat16 if device == 'gpu' else jnp.float32

                # Instantiate pipeline
                logging.info(f"Loading whisper-jax model '{model_id}' with dtype={dtype}")
                pipeline = FlaxWhisperPipline(model_id, dtype=dtype)
                
                # Transcribe
                logging.info("Starting transcription with whisper-jax")
                start_time_perf = time.time()

                # Transcribe the audio
                text = pipeline(audio_input)

                end_time_perf = time.time()

                # Output transcription
                print(text['text'], flush=True)

                # Output metrics
                transcription_time = end_time_perf - start_time_perf
                audio_file_size = os.path.getsize(audio_input) / (1024 * 1024)
                metrics_output = (
                    f"Transcription time: {transcription_time:.2f} seconds\n"
                    f"Audio file size: {audio_file_size:.2f} MB\n"
                )
                print(metrics_output, flush=True)

            except ImportError as e:
                logging.error(f"Failed to import required modules for whisper-jax: {str(e)}")
                logging.error("Please ensure you have installed whisper-jax and its dependencies correctly.")
                logging.error("You may need to update JAX and whisper-jax to compatible versions.")
                logging.error("Try running: pip install --upgrade jax jaxlib whisper-jax")
                sys.exit(1)
            except AttributeError as e:
                if 'NamedShape' in str(e):
                    logging.error("Encountered a NamedShape AttributeError.")
                else:
                    logging.error(f"An unexpected AttributeError occurred: {str(e)}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"An error occurred while using whisper-jax: {str(e)}")
                sys.exit(1)
        
        elif backend == 'OpenAI Whisper':
            import whisper
            from pydub import AudioSegment
            from pydub.silence import split_on_silence

            if device == 'mps':
                logging.warning("OpenAI Whisper does not currently support MPS device. Switching to CPU.")
                device = 'cpu'

            model = whisper.load_model(model_id, device=device)

            # Retrieve max_chunk_length, default to 0 if not provided
            max_chunk_length = float(args.max_chunk_length) if args.max_chunk_length else 0.0

            logging.info("Starting transcription")

            # Convert audio to WAV
            audio_input = convert_audio_to_wav(audio_input)

            start_oaw_time = time.time()

            if max_chunk_length > 0:
                # Load audio
                audio_segment = AudioSegment.from_wav(audio_input)

                # Split audio on silence
                chunks = split_on_silence(
                    audio_segment,
                    min_silence_len=500,
                    silence_thresh=audio_segment.dBFS - 14,
                    keep_silence=250
                )

                # Merge chunks to respect the max_chunk_length
                merged_chunks = []
                current_chunk = AudioSegment.empty()
                for chunk in chunks:
                    if len(current_chunk) + len(chunk) <= max_chunk_length * 1000:
                        current_chunk += chunk
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
                merged_chunks.append(current_chunk)

                total_offset = 0.0
                for chunk in merged_chunks:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                        chunk.export(temp_audio_file.name, format="wav")
                        result = model.transcribe(temp_audio_file.name, language=language)
                        for segment in result['segments']:
                            text = segment['text'].strip()
                            start = segment['start'] + total_offset
                            end = segment['end'] + total_offset
                            print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                    total_offset += len(chunk) / 1000.0  # Convert milliseconds to seconds
            else:
                # No chunking, process the entire audio file
                result = model.transcribe(audio_input, language=language)
                for segment in result['segments']:
                    text = segment['text'].strip()
                    start = segment['start']
                    end = segment['end']
                    print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)

            end_oaw_time = time.time()
            transcription_oaw_time = end_oaw_time - start_oaw_time
            print(f"Transcription time with OpenAI Whisper: {transcription_oaw_time:.2f} seconds", flush=True)


        end_time = time.time()
        transcription_time = end_time - start_time
        print(f"Transcription time: {transcription_time:.2f} seconds", flush=True)
    
    except Exception as e:
        error_msg = f"An error occurred during transcription: {str(e)}"
        logging.error(error_msg)
        sys.exit(1)

    finally:
        # Clean up temporary audio files
        if 'audio_path' in locals() and is_temp_file and os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    main()
