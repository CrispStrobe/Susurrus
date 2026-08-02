"""English translation catalog (source language).

Keys are grouped by area. Naming convention:

* ``app.*``      application chrome
* ``tab.*``      tab titles
* ``menu.*``     menu bar entries
* ``action.*``   menu actions
* ``btn.*``      buttons
* ``label.*``    field labels
* ``ph.*``       placeholder text
* ``tip.*``      tooltips
* ``chk.*``      checkboxes
* ``msg.*``      dialog titles (``.title``) and bodies (``.body``)
* ``help.*``     long-form help text (HTML)
* ``consent.*``  legally-operative consent text — always translate these
* ``warn.*`` / ``status.*`` / ``disclosure.*``  compliance messaging
"""

STRINGS = {
    # --- Application chrome -------------------------------------------------
    "app.title": "Susurrus",
    "app.subtitle": "Audio Transcription, TTS & Translation Suite",
    "app.tagline": "Transcribe, synthesize, and translate audio using multiple backends.",
    "app.logs_title": "Susurrus Logs",
    "app.window_subtitle": "Audio Transcription & Speech",
    # --- Tabs ---------------------------------------------------------------
    "tab.transcription": "Transcription",
    "tab.tts": "Text-to-Speech",
    "tab.translation": "Translation",
    "tab.history": "History",
    # --- Menus --------------------------------------------------------------
    "menu.file": "&File",
    "menu.tools": "&Tools",
    "menu.view": "&View",
    "menu.help": "&Help",
    "menu.language": "&Language",
    # --- Menu actions -------------------------------------------------------
    "action.open_audio": "&Open Audio File...",
    "action.save_transcript": "&Save Transcript...",
    "action.exit": "E&xit",
    "action.transcribe": "&Transcribe",
    "action.abort": "&Abort Transcription",
    "action.synthesize": "&Synthesize (TTS)",
    "action.translate": "Trans&late",
    "action.check_deps": "Check &Dependencies...",
    "action.install_deps": "&Install Dependencies...",
    "action.cuda_diagnostics": "CUDA &Diagnostics...",
    "action.install_ytdlp": "Install &yt-dlp Dependencies...",
    "action.install_voxtral": "Install &Voxtral Dependencies...",
    "action.voice_clone_wizard": "Voice &Clone Wizard...",
    "action.toggle_server": "Start/Stop &Server",
    "action.toggle_theme": "Toggle &Light/Dark Theme",
    "action.history_tab": "&History Tab",
    "action.show_logs": "Show &Logs",
    "action.about": "&About Susurrus",
    "action.diarization_help": "Speaker &Diarization Help",
    "action.audit_log": "Biometric &Audit Log...",
    # --- Buttons ------------------------------------------------------------
    "btn.transcribe": "Transcribe",
    "btn.abort": "Abort",
    "btn.save": "Save",
    "btn.browse": "Browse",
    "btn.browse_ellipsis": "Browse...",
    "btn.stream_mic": "Stream Mic",
    "btn.stop_mic": "Stop Mic",
    "btn.detect_watermark": "Detect Watermark",
    "btn.synthesize": "Synthesize",
    "btn.play": "Play",
    "btn.translate": "Translate",
    "btn.refresh": "Refresh",
    "btn.load": "Load",
    "btn.delete": "Delete",
    "btn.remove": "Remove",
    "btn.clear": "Clear",
    "btn.clear_all": "Clear All",
    "btn.close": "Close",
    "btn.cancel": "Cancel",
    "btn.back": "Back",
    "btn.next": "Next",
    "btn.clone_voice": "Clone Voice",
    "btn.add_files": "Add Files...",
    "btn.start_batch": "Start Batch",
    "btn.stop": "Stop",
    "btn.clear_done": "Clear Done",
    "btn.install_ffmpeg": "Install FFMPEG",
    "btn.install_pytorch_cuda": "Install PyTorch with CUDA support",
    "btn.install_pydub": "Install pydub",
    "btn.install_diarization": "Install Diarization Dependencies",
    "btn.install_ytdlp": "Install yt-dlp Dependencies",
    "btn.install_voxtral": "Install Voxtral Dependencies",
    # --- Labels -------------------------------------------------------------
    "label.metrics": "Metrics",
    "label.transcription": "Transcription",
    "label.audio_file": "Audio File:",
    "label.or_url": "or URL:",
    "label.backend": "Backend:",
    "label.sub_backend": "Sub-backend:",
    "label.model": "Model:",
    "label.device": "Device:",
    "label.language": "Language:",
    "label.voice": "Voice:",
    "label.tts_backend": "TTS Backend:",
    "label.output_file": "Output file:",
    "label.output_format": "Output Format:",
    "label.status": "Status:",
    "label.search": "Search history...",
    "label.batch_queue": "Batch Queue",
    "label.text_to_synthesize": "Text to synthesize:",
    "label.load_text_from_file": "Or load text from file:",
    "label.reference_audio": "Reference audio:",
    "label.reference_text": "Reference text:",
    "label.g2p_dict": "G2P dict:",
    "label.c2pa_cert": "C2PA cert:",
    "label.key": "Key:",
    "label.source_text": "Source text:",
    "label.from": "From:",
    "label.to": "To:",
    "label.translation": "Translation:",
    "label.proxy_url": "Proxy URL:",
    "label.username": "Username:",
    "label.password": "Password:",
    "label.max_chunk_length": "Max Chunk Length:",
    "label.start_time": "Start Time (s):",
    "label.end_time": "End Time (s):",
    "label.vad_model": "VAD model:",
    "label.threshold": "Threshold:",
    "label.temperature": "Temperature:",
    "label.beam_size": "Beam size:",
    "label.best_of": "Best of:",
    "label.seed": "Seed:",
    "label.method": "Method:",
    "label.max_speakers": "Max speakers:",
    "label.min_speakers": "Min. Speakers:",
    "label.max_speakers_full": "Max. Speakers:",
    "label.lid_backend": "LID backend:",
    "label.punc_model": "Punc model:",
    "label.aligner": "Aligner:",
    "label.hotwords": "Hotwords:",
    "label.boost": "Boost:",
    "label.initial_prompt": "Initial prompt:",
    "label.hf_token": "Hugging Face Token:",
    "label.diarization_model": "Diarization Model:",
    "label.mistral_api_key": "Mistral API Key:",
    "label.file_count": "{count} files",
    "label.batch_progress": "{done}/{total} done",
    "label.token_from_env": "Token loaded from environment variable",
    "label.api_key_from_env": "API key loaded from environment variable",
    "label.using_hf_token_env": "Using HF_TOKEN from environment",
    "label.using_mistral_key_env": "Using MISTRAL_API_KEY from environment",
    # --- Placeholders -------------------------------------------------------
    "ph.audio_file": "Select (or Drop) Audio file",
    "ph.url": "Or Enter URL of audio file or video link",
    "ph.tts_text": "Enter text here, or load from a file below...",
    "ph.text_file": "TXT, MD, HTML, PDF, or EPUB",
    "ph.language_code": "de",
    "ph.reference_audio": "WAV file for voice cloning (optional)",
    "ph.reference_text": "What is spoken in the reference audio (optional)",
    "ph.bundled_default": "(bundled default)",
    "ph.translate_text": "Enter text to translate...",
    "ph.proxy_url": "Enter proxy URL if needed",
    "ph.proxy_user": "Proxy username (optional)",
    "ph.proxy_pass": "Proxy password (optional)",
    "ph.max_chunk": "Max Chunk Length (seconds, 0=No Chunking)",
    "ph.start_time": "Start Time (seconds)",
    "ph.end_time": "End Time (seconds)",
    "ph.none": "(none)",
    "ph.hotwords": "comma-separated, e.g. Tokyo,CrispASR (parakeet/qwen3/voxtral)",
    "ph.initial_prompt": "Optional initial prompt for decoding",
    "ph.hf_token": "Enter your Hugging Face API token",
    "ph.auto": "Auto",
    "ph.search_logs": "Search logs...",
    "ph.mistral_key": "Enter your Mistral AI API key (for voxtral-api)",
    "ph.ref_wav_path": "Path to reference .wav file",
    "ph.ref_transcription": "Enter the transcription of the reference audio...",
    # --- Checkboxes ---------------------------------------------------------
    "chk.enable_vad": "Enable VAD",
    "chk.diarize": "Diarize",
    "chk.detect_language": "Detect language",
    "chk.split_on_punct": "Split on punct",
    "chk.flash_attention": "Flash attention",
    "chk.auto_download": "Auto-download",
    "chk.translate_to_en": "Translate to EN",
    "chk.enable_diarization": "Enable Speaker Diarization",
    "chk.no_spoken_disclaimer": "No spoken disclaimer",
    "chk.disable_watermark": "Disable watermark",
    "chk.disable_c2pa": "Disable C2PA signing",
    "chk.accept_marking_responsibility": "Accept marking responsibility",
    # --- Tooltips -----------------------------------------------------------
    "tip.save_transcription": "Save Transcription",
    "tip.stream_mic": "Live transcription from microphone",
    "tip.detect_watermark": "Check if the loaded audio contains an AI-generated watermark",
    "tip.load_entry": "Load selected transcript into output",
    "tip.delete_entry": "Delete selected entry",
    "tip.clear_all_history": "Delete all history entries",
    "tip.enable_diarization": "Identify different speakers in the audio",
    "tip.hf_token": (
        "Required for speaker diarization. " "Get it from https://huggingface.co/settings/tokens"
    ),
    "tip.mistral_key": (
        "Required for voxtral-api backend.\n" "Get your API key from: https://console.mistral.ai/"
    ),
    "tip.edited": "Edited",
    "tip.confidence": "Confidence: {label}",
    "tip.no_spoken_disclaimer": (
        "Skip the audible AI-disclosure prefix. Machine-readable marking "
        "(watermark, AI marker, C2PA) is still applied."
    ),
    "tip.disable_watermark": (
        "Disable the neural AI-content watermark. The declarative AI marker "
        "still applies (EU AI Act Art. 50)."
    ),
    "tip.disable_c2pa": (
        "Skip C2PA Content Credentials. The declarative AI marker is still "
        "embedded unless marking responsibility is accepted."
    ),
    "tip.accept_marking_responsibility": (
        "Emit completely unmarked audio. You take on responsibility for "
        "marking this AI-generated content (EU AI Act Art. 50)."
    ),
    # --- Consent (legally operative — always localized) ---------------------
    "consent.clone_checkbox": "I have the right to clone this voice",
    "consent.clone_detail": (
        "I confirm this is my own voice, or that the speaker consented "
        "to having their voice cloned. Required under EU AI Act Art. 50."
    ),
    # --- Compliance messaging ----------------------------------------------
    "warn.no_watermark": (
        "Watermarking disabled. AI-content marking responsibility "
        "rests with the operator per EU AI Act Art. 50."
    ),
    "warn.marking_opted_out": (
        "AI-content marking skipped. Responsibility for marking this "
        "output rests with the operator per EU AI Act Art. 50."
    ),
    "warn.marking_failed": (
        "WARNING: could not mark this audio as AI-generated. EU AI Act "
        "Art. 50(2) requires machine-readable marking of synthetic audio."
    ),
    "status.marked": "Marked as AI-generated ({layers}).",
    "error.marking_required": (
        "Susurrus does not produce unmarked synthetic audio. Choose a .wav or "
        ".mp3 output path, install the marking stack, or tick 'I accept "
        "marking responsibility' to take the EU AI Act Art. 50 obligation on "
        "yourself."
    ),
    "warn.disclosure_missing": (
        "WARNING: this audio clones a voice but carries no audible "
        "disclosure. EU AI Act Art. 50(4) requires disclosure that deepfake "
        "content is artificially generated, and machine-readable marking does "
        "not discharge it for a listener."
    ),
    "warn.marking_unsupported_format": (
        "WARNING: this audio format carries no declarative AI marker, so the "
        "output may be unmarked. EU AI Act Art. 50(2) requires machine-readable "
        "marking of synthetic audio — use .wav or .mp3."
    ),
    "error.marking_opt_out_needs_attestation": (
        "Disabling the watermark, C2PA signing or the spoken disclaimer "
        "reduces EU AI Act Art. 50 provenance. Tick 'I accept marking "
        "responsibility' to confirm that the obligation to mark and disclose "
        "this output rests with you as the operator."
    ),
    "status.tts_initializing": "Initializing TTS backend: {backend}",
    "status.tts_synthesizing": "Synthesizing with {backend}...",
    "status.audio_saved": "Audio saved to: {path}",
    "warn.speaker_db_consent": (
        "Speaker enrollment stores voice biometrics linked to a named "
        "person. Confirm you have a lawful basis and the speaker's "
        "consent (GDPR Art. 9); pass --speaker-db-consent to attest."
    ),
    "disclosure.spoken": "The following audio was generated by artificial intelligence.",
    "notice.synthetic_text": (
        "This text was produced by a machine-translation model. It loses "
        "nuance and can invert meaning — have a person review it. If you "
        "publish it to inform the public on a matter of public interest, "
        "disclosing that it is AI-generated is your obligation (EU AI Act "
        "Art. 50(4))."
    ),
    # --- AI literacy (Art. 4) / intended purpose ---------------------------
    "action.ai_notice": "About AI in Susurrus",
    "msg.about.title": "About Susurrus",
    # --- Long-form help and status (migrated from hardcoded English) --------
    "help.hf_token": (
        "A Hugging Face API token is required for speaker diarization.\n\n"
        "1. Create a free account at https://huggingface.co\n"
        "2. Go to https://huggingface.co/settings/tokens\n"
        "3. Create a new token with 'read' access\n"
        "4. Copy and paste the token here\n\n"
        "Note: You need to accept the user agreement for the diarization "
        "models at https://huggingface.co/pyannote/speaker-diarization"
    ),
    "help.diarization_model": (
        "Choose the appropriate diarization model for your audio:\n\n"
        "• Default: General purpose diarization model\n"
        "• English: Optimized for English conversations\n"
        "• Chinese: Optimized for Mandarin Chinese conversations\n"
        "• German: Optimized for German conversations\n"
        "• Spanish: Optimized for Spanish conversations\n"
        "• Japanese: Optimized for Japanese conversations\n\n"
        "Language-specific models may provide better results for their "
        "respective languages, especially for phone calls and naturalistic "
        "conversations."
    ),
    "help.mistral_key": (
        "<h3>Mistral AI API Key</h3>"
        "<p>The Mistral API key is required for the <b>voxtral-api</b> backend.</p>"
        "<h4>How to get your API key:</h4>"
        "<ol>"
        "<li>Create a free account at "
        "<a href='https://console.mistral.ai/'>console.mistral.ai</a></li>"
        "<li>Navigate to API Keys section</li>"
        "<li>Create a new API key</li>"
        "<li>Copy and paste it here</li>"
        "</ol>"
        "<h4>Alternatively:</h4>"
        "<p>You can set the <code>MISTRAL_API_KEY</code> environment variable:</p>"
        "<p><b>PowerShell:</b> <code>$env:MISTRAL_API_KEY = 'your-key'</code></p>"
        "<p><b>CMD:</b> <code>set MISTRAL_API_KEY=your-key</code></p>"
        "<p><b>Linux/Mac:</b> <code>export MISTRAL_API_KEY='your-key'</code></p>"
    ),
    "help.diarization_unavailable": (
        "Speaker diarization is not available. Please ensure you have:\n\n"
        "1. Installed pyannote.audio\n"
        "2. Set a valid Hugging Face token in the HF_TOKEN environment variable\n\n"
        "If you still see this message, there may be a version conflict "
        "between packages."
    ),
    "label.history_entries": "{count} entries",
    "msg.delete_entry.body": "Delete '{title}'?",
    "error.delete_failed": "Failed to delete: {error}",
    "error.clear_failed": "Failed to clear: {error}",
    "error.read_file": "Error reading file: {error}",
    "error.generic": "Error: {error}",
    "error.watermark_detection_failed": "Watermark detection failed: {error}",
    "error.save_failed": "Failed to save: {error}",
    "error.dependency_check": "There was an error checking dependencies: {error}",
    "msg.backend_unavailable.body": (
        "The '{backend}' backend is not compiled into the CrispASR binary.\n\n"
        "Available backends: {available}\n\n"
        "Rebuild CrispASR with the required backend enabled."
    ),
    "msg.play_failed.body": "Could not play audio: {error}\n\nThe file is saved at: {path}",
    "msg.saved.title": "Saved",
    "msg.saved_to": "Saved to: {path}",
    "status.transcription_aborted": "Transcription aborted by user.",
    "status.starting_synthesis": "Starting synthesis...",
    "status.server_stopped": "Server stopped.",
    "status.server_started": "Server started on {host}:{port}",
    "status.server_marking_proxy": (
        "Audio responses are marked as AI-generated before they leave this "
        "process (EU AI Act Art. 50(2)). Responses that cannot be marked are "
        "refused rather than served unmarked."
    ),
    "msg.server_proxy_failed": (
        "Could not start the AI-marking proxy: {reason}.\n\n"
        "The server was not started. Without the proxy, Susurrus is not in the "
        "response path and cannot mark or verify the synthetic audio the "
        "endpoint returns, which EU AI Act Art. 50(2) requires."
    ),
    "status.tts_done": "Done! Audio saved to: {path}",
    "status.install_issues": "Installation completed with issues: {detail}",
    "msg.install_voxtral.body": (
        "This will install the development version of transformers.\n\n"
        "The following packages will be installed:\n"
        "• transformers (from GitHub)\n"
        "• mistral-common[audio]\n"
        "• soundfile\n\n"
        "This may take several minutes. Continue?"
    ),
    "msg.diarization_info.body": (
        "You are using speaker diarization for the first time.\n\n"
        "Important notes:\n"
        "- The first run will download the diarization model (approx. 1GB)\n"
        "- Processing may take longer than standard transcription\n"
        "- For language-specific content, consider using the matching "
        "language model\n\n"
        "Do you want to continue?"
    ),
    "help.ffmpeg_install": (
        "1. Download the 'essentials' build\n"
        "2. Extract the zip file to C:\\ffmpeg\n"
        "3. Add C:\\ffmpeg\\bin to your system PATH:\n"
        "   - Open Control Panel > System > Advanced System Settings\n"
        "   - Click 'Environment Variables'\n"
        "   - Edit the 'Path' variable and add C:\\ffmpeg\\bin\n"
        "   - Click OK and restart your terminal"
    ),
    "help.voxtral_install": (
        "<h3>Installing Voxtral Dependencies</h3>"
        "<p>Run these commands:</p>"
        "<p><b>Windows:</b></p>"
        "<pre>install_voxtral.bat</pre>"
        "<p><b>Linux/Mac:</b></p>"
        "<pre>./install_voxtral.sh</pre>"
        "<p><b>Or manually:</b></p>"
        "<pre>pip uninstall transformers -y\n"
        "pip install git+https://github.com/huggingface/transformers.git\n"
        "pip install mistral-common[audio] soundfile</pre>"
    ),
    "help.diarization": (
        "<h2>Speaker Diarization in Susurrus</h2>"
        "<p>Speaker diarization identifies different speakers in your audio "
        "recordings and creates transcriptions with speaker labels.</p>"
        "<h3>Methods</h3>"
        "<ul>"
        "<li><b>PyAnnote</b> — Neural model (requires HF token)</li>"
        "<li><b>CrispASR methods</b> — energy, xcorr, vad-turns, sherpa, ecapa</li>"
        "</ul>"
        "<h3>Requirements (PyAnnote)</h3>"
        "<ul>"
        "<li>A Hugging Face account and API token</li>"
        "<li>The pyannote.audio library installed</li>"
        "<li>Acceptance of the model license agreements</li>"
        "</ul>"
        "<h3>Tips for best results</h3>"
        "<ul>"
        "<li>Use clean audio with minimal background noise</li>"
        "<li>Choose language-specific models for non-English content</li>"
        "<li>Set min/max speakers if you know how many speakers to expect</li>"
        "</ul>"
    ),
    "msg.about.body": (
        "<h1>{name}</h1>"
        "<p>Audio Transcription, TTS, Translation &amp; S2S Suite</p>"
        "<p>Version {version}</p>"
        "<p>Features:</p>"
        "<ul>"
        "<li>38+ ASR backends via CrispASR</li>"
        "<li>27+ TTS engines (local and cloud)</li>"
        "<li>Multi-language translation (m2m100, MadLad, Gemma4)</li>"
        "<li>Speech-to-speech (lfm2-audio, mini-omni2)</li>"
        "<li>Speaker diarization (PyAnnote + CrispASR methods)</li>"
        "<li>Export: SRT, VTT, JSON, CSV, TXT</li>"
        "<li>Transcription history with search</li>"
        "<li>Batch processing queue</li>"
        "<li>Light/dark themes</li>"
        "<li>Waveform display, real-time progress</li>"
        "<li>Standalone alignment (--align-only)</li>"
        "</ul>"
        "<p>Shortcuts: F5=Transcribe, Ctrl+S=Save, Ctrl+T=Theme, Ctrl+H=History</p>"
        "<p>Synthesized audio is marked as AI-generated — see "
        "<i>Help &gt; About AI in Susurrus</i>.</p>"
    ),
    "msg.ai_notice.title": "About AI in Susurrus",
    "msg.ai_notice.body": (
        "<h3>This is an AI system</h3>"
        "<p>Susurrus transcribes, synthesizes, translates and separates "
        "speakers using machine-learning models. Everything it produces is a "
        "<b>model prediction, not a record</b>.</p>"
        "<h4>Intended purpose</h4>"
        "<p>A local-first tool for processing your own or consented material, "
        "with a person reviewing the output before it is relied on.</p>"
        "<h4>Known limitations</h4>"
        "<ul>"
        "<li><b>Transcription</b> contains errors. Accuracy varies sharply "
        "with accent, audio quality, background noise, domain vocabulary and "
        "language; non-native accents and under-resourced languages typically "
        "fare worse.</li>"
        "<li><b>Diarization</b> guesses speaker boundaries and counts. "
        "Overlapping speech, similar voices and short turns fail often.</li>"
        "<li><b>Translation</b> loses nuance and can invert meaning, "
        "especially around negation, idiom and ambiguous pronouns.</li>"
        "<li><b>Synthesized speech</b> is not evidence that any real person "
        "said anything.</li>"
        "</ul>"
        "<h4>Not validated for</h4>"
        "<p>Any use where an error carries legal or safety consequences "
        "without human review: evidentiary transcripts, medical "
        "documentation, employment or education decisions, credit or benefit "
        "eligibility, law enforcement, migration and border control.</p>"
        "<h4>Your obligations</h4>"
        "<p>Synthetic audio is marked as AI-generated automatically. "
        "Disclosing to the people who see or hear the output that it is "
        "artificially generated remains yours to do. See COMPLIANCE.md.</p>"
    ),
    "warn.no_transcription": "No transcription available to save.",
    # --- Wizard -------------------------------------------------------------
    "wizard.title": "Voice Clone Wizard",
    "wizard.step1_title": "Step 1: Select Reference Audio",
    "wizard.step1_body": "Choose a .wav file of the voice you want to clone.",
    "wizard.step2_title": "Step 2: Reference Transcription",
    "wizard.step2_body": (
        "Enter what is spoken in the reference audio. " "This helps the TTS engine match the voice."
    ),
    "wizard.step3_title": "Step 3: Confirm",
    "wizard.step3_body": "Voice cloning requires consent attestation (EU AI Act).",
    "wizard.select_audio_dialog": "Select Reference Audio",
    "wizard.summary_audio": "Audio:",
    "wizard.summary_ref_text": "Ref text:",
    # --- Dialog titles ------------------------------------------------------
    "msg.error.title": "Error",
    "msg.warning.title": "Warning",
    "msg.success.title": "Success",
    "msg.timeout.title": "Timeout",
    # --- Dialog messages ----------------------------------------------------
    "msg.no_audio_input.title": "No Audio Input",
    "msg.no_audio_input.body": "Please provide either an audio file or URL.",
    "msg.backend_unavailable.title": "Backend Not Available",
    "msg.unsupported_url.title": "Unsupported URL",
    "msg.unsupported_url.body": (
        "Downloading from YouTube is not supported due to Terms of Service "
        "restrictions.\n\nPlease use a local file or a direct URL to a "
        "non-YouTube media file."
    ),
    "msg.missing_api_key.title": "Missing API Key",
    "msg.missing_api_key.body": (
        "Mistral AI API key is required for voxtral-api backend.\n\n"
        "Please enter your API key in the Voxtral API Settings section, "
        "or set the MISTRAL_API_KEY environment variable."
    ),
    "msg.no_text.title": "No Text",
    "msg.no_text_tts.body": "Please enter text or load a text file.",
    "msg.no_text_translate.body": "Please enter text to translate.",
    "msg.tts_error.title": "TTS Error",
    "msg.no_audio.title": "No Audio",
    "msg.no_audio.body": "No TTS output available to play.",
    "msg.playback_error.title": "Playback Error",
    "msg.translation_error.title": "Translation Error",
    "msg.translating": "Translating...",
    "msg.translation_complete": "Translation complete.",
    "msg.crispasr_missing.body": "CrispASR binary not found.",
    "msg.crispasr_missing_watermark.body": ("CrispASR binary not found for watermark detection."),
    "msg.no_audio_selected.body": "No audio file selected.",
    "msg.watermark_detection.title": "Watermark Detection",
    "msg.watermark_timeout.body": "Watermark detection timed out.",
    "msg.dependency_check_error.title": "Dependency Check Error",
    "msg.ffmpeg_missing.title": "FFMPEG Not Found",
    "msg.ffmpeg_missing.body": (
        "FFMPEG is not installed or not in your PATH. Audio format support "
        "will be limited.\n\nYou can install FFMPEG through the "
        "Tools > Install Dependencies menu."
    ),
    "msg.cuda_missing.title": "PyTorch CUDA Support Missing",
    "msg.cuda_missing.body": (
        "An NVIDIA GPU was detected, but PyTorch was installed without CUDA "
        "support.\n\nWould you like to reinstall PyTorch with CUDA support now?"
    ),
    "msg.voxtral_missing.title": "Voxtral Dependencies Missing",
    "msg.voxtral_missing.body": (
        "The voxtral-local backend requires the development version of "
        "transformers.\n\nWould you like to view installation instructions?"
    ),
    "msg.diarization_unavailable.title": "Diarization Not Available",
    "msg.missing_token.title": "Missing Token",
    "msg.missing_token.body": (
        "A Hugging Face token is required for speaker diarization.\n\n"
        "Please enter your token or disable speaker diarization."
    ),
    "msg.diarization_info.title": "Speaker Diarization Information",
    "msg.delete_entry.title": "Delete Entry",
    "msg.clear_history.title": "Clear History",
    "msg.clear_history.body": "Delete ALL history entries? This cannot be undone.",
    "msg.select_ref_audio.body": "Please select a reference audio file.",
    "msg.install_deps.title": "Install Dependencies",
    "msg.install_complete": "Installation completed successfully!",
    "msg.preparing": "Preparing...",
    "msg.install_pytorch_cuda.title": "Install PyTorch with CUDA",
    "msg.install_pytorch_cuda.body": ("This will reinstall PyTorch with CUDA support. Continue?"),
    "msg.install_voxtral.title": "Install Voxtral Dependencies",
    "msg.ffmpeg_instructions.title": "FFMPEG Installation Instructions",
    "msg.hf_token_help.title": "Hugging Face Token Help",
    "msg.diarization_model_help.title": "Diarization Model Selection",
    "msg.mistral_key_help.title": "Mistral API Key Help",
    "msg.voxtral_instructions.title": "Voxtral Installation Instructions",
    "msg.diarization_help.title": "Speaker Diarization Help",
    "msg.language_changed.title": "Language Changed",
    "msg.language_changed.body": (
        "The interface language has been changed. Restart Susurrus to apply " "it everywhere."
    ),
    "msg.audit_log.title": "Biometric Audit Log",
    "msg.audit_log_empty": "No biometric events have been recorded.",
    "msg.audit_log_valid": "Audit chain verified: {count} events, no tampering detected.",
    "msg.audit_log_invalid": "Audit chain BROKEN: {errors}",
    # --- Dialog headings ----------------------------------------------------
    "heading.dependency_status": "Dependency Status",
    "heading.dependencies_required": (
        "The following dependencies are required or recommended for Susurrus:"
    ),
    "heading.install_instructions": "Installation Instructions:",
    "help.install_pip": (
        "Missing required dependencies can be installed with pip:<br>"
        "<code>pip install torch transformers pydub numpy</code><br><br>"
        "For speaker diarization, also install:<br>"
        "<code>pip install pyannote.audio huggingface_hub</code><br><br>"
        "For ffmpeg, visit "
        "<a href='https://ffmpeg.org/download.html'>ffmpeg.org/download.html</a>"
    ),
    "heading.install_missing": "Install Missing Dependencies",
    "heading.deps_needed": "The following dependencies are needed for full functionality:",
    "heading.cuda_diagnostics": "CUDA and NVIDIA Diagnostics",
    "dialog.dependencies.title": "Susurrus Dependencies",
    # --- Voxtral panel ------------------------------------------------------
    "voxtral.description": (
        "Voxtral is Mistral AI's speech recognition model.\n"
        "It supports 8 languages and offers both local and API-based inference."
    ),
    "voxtral.languages": (
        "<b>Supported Languages:</b> English, French, Spanish, German, "
        "Italian, Portuguese, Polish, Dutch"
    ),
    "voxtral.note": (
        "<b>Note:</b> voxtral-local requires transformers from GitHub. "
        "Run install_voxtral.sh/bat to set up."
    ),
}
