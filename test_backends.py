# test_backends.py
from workers.transcription.backends import get_backend

def test_backend(backend_name, audio_file):
    backend = get_backend(
        backend_name,
        model_id='tiny',
        device='cpu',
        language='en'
    )
    
    for start, end, text in backend.transcribe(audio_file):
        print(f"[{start:.3f} -> {end:.3f}] {text}")