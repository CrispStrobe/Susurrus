# workers/transcription/utils.py
"""Transcription utilities"""
import sys
import io

# Set up UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class TranscriptionUtils:
    """Utility functions for transcription"""
    
    @staticmethod
    def setup_environment():
        """Set up environment for transcription"""
        # [COPY environment setup from transcribe_worker.py]
        pass
    
    @staticmethod
    def parse_arguments():
        """Parse command line arguments"""
        import argparse
        
        parser = argparse.ArgumentParser(description='Susurrus Transcription worker')
        parser.add_argument('--audio-input', help='Path to audio input')
        parser.add_argument('--audio-url', help='URL to audio file')
        parser.add_argument('--model-id', type=str, default=None)
        parser.add_argument('--word-timestamps', action='store_true')
        parser.add_argument('--language', default=None)
        parser.add_argument('--backend', type=str, default='mlx-whisper')
        parser.add_argument('--device', type=str, default='auto')
        parser.add_argument('--pipeline-type', default='default')
        parser.add_argument('--max-chunk-length', type=float, default=0.0)
        parser.add_argument('--output-format', default='txt')
        parser.add_argument('--quantization', default=None)
        parser.add_argument('--preprocessor-path', type=str, default='')
        parser.add_argument('--original-model-id', type=str, default='')
        parser.add_argument('--start-time', type=str, default=None)
        parser.add_argument('--end-time', type=str, default=None)
        parser.add_argument('--proxy-url', type=str, default=None)
        parser.add_argument('--proxy-username', type=str, default=None)
        parser.add_argument('--proxy-password', type=str, default=None)
        parser.add_argument('--ffmpeg-path', type=str, default='ffmpeg')
        parser.add_argument('--mistral-api-key', type=str, default=None)
        parser.add_argument('--temperature', type=float, default=0.0)
        
        return parser.parse_args()
    
    @staticmethod
    def detect_device(device_arg):
        """Detect and validate device"""
        try:
            import torch
            if device_arg == 'auto':
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            elif device_arg == 'cpu':
                return "cpu"
            elif device_arg == 'gpu':
                return "cuda"
            elif device_arg == 'mps':
                return "mps"
            else:
                return "cpu"
        except ImportError:
            logging.warning("PyTorch not installed, defaulting to CPU")
            return "cpu"
    
    @staticmethod
    def setup_proxies(proxy_url, proxy_username, proxy_password):
        """Set up proxy configuration"""
        if not proxy_url:
            return None
        
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        
        if proxy_username and proxy_password:
            from urllib.parse import urlparse, urlunparse
            parsed_url = urlparse(proxy_url)
            netloc = f"{proxy_username}:{proxy_password}@{parsed_url.hostname}"
            if parsed_url.port:
                netloc += f":{parsed_url.port}"
            proxy_url_with_auth = urlunparse((
                parsed_url.scheme,
                netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment
            ))
            proxies['http'] = proxy_url_with_auth
            proxies['https'] = proxy_url_with_auth
        
        return proxies