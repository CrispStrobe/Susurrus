"""German translation catalog.

Every key in ``en.py`` must exist here — ``tests/unit/test_i18n.py`` enforces
it, so a forgotten translation fails the suite instead of silently showing
English to a German user. That matters most for ``consent.*`` and the
compliance messaging, which are legally operative text.
"""

STRINGS = {
    # --- Application chrome -------------------------------------------------
    "app.title": "Susurrus",
    "app.subtitle": "Audio-Transkription, TTS & Übersetzungssuite",
    "app.tagline": ("Audio transkribieren, synthetisieren und übersetzen mit mehreren Backends."),
    "app.logs_title": "Susurrus-Protokoll",
    "app.window_subtitle": "Audiotranskription & Sprachsynthese",
    # --- Tabs ---------------------------------------------------------------
    "tab.transcription": "Transkription",
    "tab.tts": "Sprachsynthese",
    "tab.translation": "Übersetzung",
    "tab.history": "Verlauf",
    # --- Menus --------------------------------------------------------------
    "menu.file": "&Datei",
    "menu.tools": "&Werkzeuge",
    "menu.view": "&Ansicht",
    "menu.help": "&Hilfe",
    "menu.language": "&Sprache",
    # --- Menu actions -------------------------------------------------------
    "action.open_audio": "Audiodatei &öffnen...",
    "action.save_transcript": "Transkript &speichern...",
    "action.exit": "&Beenden",
    "action.transcribe": "&Transkribieren",
    "action.abort": "Transkription &abbrechen",
    "action.synthesize": "&Sprachsynthese (TTS)",
    "action.translate": "Über&setzen",
    "action.check_deps": "&Abhängigkeiten prüfen...",
    "action.install_deps": "Abhängigkeiten &installieren...",
    "action.cuda_diagnostics": "CUDA-&Diagnose...",
    "action.install_ytdlp": "&yt-dlp-Abhängigkeiten installieren...",
    "action.install_voxtral": "&Voxtral-Abhängigkeiten installieren...",
    "action.voice_clone_wizard": "Stimm&klon-Assistent...",
    "action.toggle_server": "Server &starten/stoppen",
    "action.toggle_theme": "&Helles/dunkles Design umschalten",
    "action.history_tab": "&Verlauf",
    "action.show_logs": "&Protokoll anzeigen",
    "action.about": "Ü&ber Susurrus",
    "action.diarization_help": "Hilfe zur Sprecher&trennung",
    "action.audit_log": "Biometrisches &Prüfprotokoll...",
    # --- Buttons ------------------------------------------------------------
    "btn.transcribe": "Transkribieren",
    "btn.abort": "Abbrechen",
    "btn.save": "Speichern",
    "btn.browse": "Durchsuchen",
    "btn.browse_ellipsis": "Durchsuchen...",
    "btn.stream_mic": "Mikrofon streamen",
    "btn.stop_mic": "Mikrofon stoppen",
    "btn.detect_watermark": "Wasserzeichen erkennen",
    "btn.synthesize": "Synthetisieren",
    "btn.play": "Abspielen",
    "btn.translate": "Übersetzen",
    "btn.refresh": "Aktualisieren",
    "btn.load": "Laden",
    "btn.delete": "Löschen",
    "btn.remove": "Entfernen",
    "btn.clear": "Leeren",
    "btn.clear_all": "Alle löschen",
    "btn.close": "Schließen",
    "btn.cancel": "Abbrechen",
    "btn.back": "Zurück",
    "btn.next": "Weiter",
    "btn.clone_voice": "Stimme klonen",
    "btn.add_files": "Dateien hinzufügen...",
    "btn.start_batch": "Stapel starten",
    "btn.stop": "Stoppen",
    "btn.clear_done": "Erledigte löschen",
    "btn.install_ffmpeg": "FFMPEG installieren",
    "btn.install_pytorch_cuda": "PyTorch mit CUDA-Unterstützung installieren",
    "btn.install_pydub": "pydub installieren",
    "btn.install_diarization": "Abhängigkeiten für Sprechertrennung installieren",
    "btn.install_ytdlp": "yt-dlp-Abhängigkeiten installieren",
    "btn.install_voxtral": "Voxtral-Abhängigkeiten installieren",
    # --- Labels -------------------------------------------------------------
    "label.metrics": "Metriken",
    "label.transcription": "Transkription",
    "label.audio_file": "Audiodatei:",
    "label.or_url": "oder URL:",
    "label.backend": "Backend:",
    "label.sub_backend": "Sub-Backend:",
    "label.model": "Modell:",
    "label.device": "Gerät:",
    "label.language": "Sprache:",
    "label.voice": "Stimme:",
    "label.tts_backend": "TTS-Backend:",
    "label.output_file": "Ausgabedatei:",
    "label.output_format": "Ausgabeformat:",
    "label.status": "Status:",
    "label.search": "Verlauf durchsuchen...",
    "label.batch_queue": "Warteschlange",
    "label.text_to_synthesize": "Zu synthetisierender Text:",
    "label.load_text_from_file": "Oder Text aus Datei laden:",
    "label.reference_audio": "Referenzaudio:",
    "label.reference_text": "Referenztext:",
    "label.g2p_dict": "G2P-Wörterbuch:",
    "label.c2pa_cert": "C2PA-Zertifikat:",
    "label.key": "Schlüssel:",
    "label.source_text": "Quelltext:",
    "label.from": "Von:",
    "label.to": "Nach:",
    "label.translation": "Übersetzung:",
    "label.proxy_url": "Proxy-URL:",
    "label.username": "Benutzername:",
    "label.password": "Passwort:",
    "label.max_chunk_length": "Max. Blocklänge:",
    "label.start_time": "Startzeit (s):",
    "label.end_time": "Endzeit (s):",
    "label.vad_model": "VAD-Modell:",
    "label.threshold": "Schwellenwert:",
    "label.temperature": "Temperatur:",
    "label.beam_size": "Beam-Breite:",
    "label.best_of": "Best-of:",
    "label.seed": "Startwert:",
    "label.method": "Methode:",
    "label.max_speakers": "Max. Sprecher:",
    "label.min_speakers": "Min. Sprecher:",
    "label.max_speakers_full": "Max. Sprecher:",
    "label.lid_backend": "LID-Backend:",
    "label.punc_model": "Interpunktionsmodell:",
    "label.aligner": "Aligner:",
    "label.hotwords": "Schlüsselwörter:",
    "label.boost": "Verstärkung:",
    "label.initial_prompt": "Anfangs-Prompt:",
    "label.hf_token": "Hugging-Face-Token:",
    "label.diarization_model": "Modell zur Sprechertrennung:",
    "label.mistral_api_key": "Mistral-API-Schlüssel:",
    "label.file_count": "{count} Dateien",
    "label.batch_progress": "{done}/{total} erledigt",
    "label.token_from_env": "Token aus Umgebungsvariable geladen",
    "label.api_key_from_env": "API-Schlüssel aus Umgebungsvariable geladen",
    "label.using_hf_token_env": "HF_TOKEN aus der Umgebung wird verwendet",
    "label.using_mistral_key_env": "MISTRAL_API_KEY aus der Umgebung wird verwendet",
    # --- Placeholders -------------------------------------------------------
    "ph.audio_file": "Audiodatei auswählen (oder hierher ziehen)",
    "ph.url": "Oder URL einer Audiodatei bzw. eines Videos eingeben",
    "ph.tts_text": "Text hier eingeben oder unten aus einer Datei laden...",
    "ph.text_file": "TXT, MD, HTML, PDF oder EPUB",
    "ph.language_code": "de",
    "ph.reference_audio": "WAV-Datei zum Klonen der Stimme (optional)",
    "ph.reference_text": "Was im Referenzaudio gesprochen wird (optional)",
    "ph.bundled_default": "(mitgelieferte Standardwerte)",
    "ph.translate_text": "Zu übersetzenden Text eingeben...",
    "ph.proxy_url": "Bei Bedarf Proxy-URL eingeben",
    "ph.proxy_user": "Proxy-Benutzername (optional)",
    "ph.proxy_pass": "Proxy-Passwort (optional)",
    "ph.max_chunk": "Max. Blocklänge (Sekunden, 0 = keine Aufteilung)",
    "ph.start_time": "Startzeit (Sekunden)",
    "ph.end_time": "Endzeit (Sekunden)",
    "ph.none": "(keiner)",
    "ph.hotwords": "kommagetrennt, z. B. Tokio,CrispASR (parakeet/qwen3/voxtral)",
    "ph.initial_prompt": "Optionaler Anfangs-Prompt für die Dekodierung",
    "ph.hf_token": "Hugging-Face-API-Token eingeben",
    "ph.auto": "Automatisch",
    "ph.search_logs": "Protokoll durchsuchen...",
    "ph.mistral_key": "Mistral-AI-API-Schlüssel eingeben (für voxtral-api)",
    "ph.ref_wav_path": "Pfad zur Referenz-WAV-Datei",
    "ph.ref_transcription": "Transkription des Referenzaudios eingeben...",
    # --- Checkboxes ---------------------------------------------------------
    "chk.enable_vad": "VAD aktivieren",
    "chk.diarize": "Sprecher trennen",
    "chk.detect_language": "Sprache erkennen",
    "chk.split_on_punct": "Bei Satzzeichen trennen",
    "chk.flash_attention": "Flash Attention",
    "chk.auto_download": "Automatisch herunterladen",
    "chk.translate_to_en": "Ins Englische übersetzen",
    "chk.enable_diarization": "Sprechertrennung aktivieren",
    "chk.no_spoken_disclaimer": "Kein gesprochener Hinweis",
    "chk.disable_watermark": "Wasserzeichen deaktivieren",
    "chk.disable_c2pa": "C2PA-Signatur deaktivieren",
    "chk.accept_marking_responsibility": "Kennzeichnungsverantwortung übernehmen",
    # --- Tooltips -----------------------------------------------------------
    "tip.save_transcription": "Transkription speichern",
    "tip.stream_mic": "Live-Transkription vom Mikrofon",
    "tip.detect_watermark": ("Prüfen, ob das geladene Audio ein KI-Wasserzeichen enthält"),
    "tip.load_entry": "Ausgewähltes Transkript in die Ausgabe laden",
    "tip.delete_entry": "Ausgewählten Eintrag löschen",
    "tip.clear_all_history": "Alle Verlaufseinträge löschen",
    "tip.enable_diarization": "Verschiedene Sprecher im Audio unterscheiden",
    "tip.hf_token": (
        "Für die Sprechertrennung erforderlich. "
        "Erhältlich unter https://huggingface.co/settings/tokens"
    ),
    "tip.mistral_key": (
        "Für das voxtral-api-Backend erforderlich.\n"
        "API-Schlüssel erhältlich unter: https://console.mistral.ai/"
    ),
    "tip.edited": "Bearbeitet",
    "tip.confidence": "Konfidenz: {label}",
    "tip.no_spoken_disclaimer": (
        "Den hörbaren KI-Hinweis überspringen. Die maschinenlesbare "
        "Kennzeichnung (Wasserzeichen, KI-Marker, C2PA) bleibt bestehen."
    ),
    "tip.disable_watermark": (
        "Das neuronale KI-Wasserzeichen deaktivieren. Der deklarative "
        "KI-Marker bleibt bestehen (EU AI Act Art. 50)."
    ),
    "tip.disable_c2pa": (
        "C2PA Content Credentials überspringen. Der deklarative KI-Marker "
        "wird weiterhin eingebettet, sofern die Kennzeichnungsverantwortung "
        "nicht übernommen wird."
    ),
    "tip.accept_marking_responsibility": (
        "Vollständig ungekennzeichnetes Audio ausgeben. Sie übernehmen die "
        "Verantwortung für die Kennzeichnung dieser KI-generierten Inhalte "
        "(EU AI Act Art. 50)."
    ),
    # --- Consent (legally operative — always localized) ---------------------
    "consent.clone_checkbox": "Ich habe das Recht, diese Stimme zu klonen",
    "consent.clone_detail": (
        "Ich bestätige, dass es sich um meine eigene Stimme handelt oder "
        "die sprechende Person dem Klonen ihrer Stimme zugestimmt hat. "
        "Erforderlich nach EU AI Act Art. 50."
    ),
    # --- Compliance messaging ----------------------------------------------
    "warn.no_watermark": (
        "Wasserzeichen deaktiviert. Die Verantwortung für die "
        "KI-Inhaltskennzeichnung liegt beim Betreiber (EU AI Act Art. 50)."
    ),
    "warn.marking_opted_out": (
        "KI-Inhaltskennzeichnung übersprungen. Die Verantwortung für die "
        "Kennzeichnung dieser Ausgabe liegt beim Betreiber (EU AI Act "
        "Art. 50)."
    ),
    "warn.marking_failed": (
        "WARNUNG: Diese Audiodatei konnte nicht als KI-generiert "
        "gekennzeichnet werden. EU AI Act Art. 50(2) verlangt eine "
        "maschinenlesbare Kennzeichnung synthetischer Audioinhalte."
    ),
    "status.marked": "Als KI-generiert gekennzeichnet ({layers}).",
    "error.marking_required": (
        "Susurrus erzeugt keine ungekennzeichneten synthetischen Audioinhalte. "
        "Wählen Sie einen .wav- oder .mp3-Ausgabepfad, installieren Sie die "
        "Kennzeichnungskomponenten, oder aktivieren Sie „Ich übernehme die "
        "Kennzeichnungsverantwortung“, um die Pflicht nach EU AI Act Art. 50 "
        "selbst zu übernehmen."
    ),
    "warn.disclosure_missing": (
        "WARNUNG: Diese Audiodatei klont eine Stimme, enthält aber keinen "
        "hörbaren Hinweis. EU AI Act Art. 50(4) verlangt die Offenlegung, "
        "dass Deepfake-Inhalte künstlich erzeugt wurden; eine maschinen"
        "lesbare Kennzeichnung erfüllt diese Pflicht gegenüber Hörenden nicht."
    ),
    "warn.marking_unsupported_format": (
        "WARNUNG: Dieses Audioformat trägt keine deklarative "
        "KI-Kennzeichnung, die Ausgabe ist daher möglicherweise ungekenn"
        "zeichnet. EU AI Act Art. 50(2) verlangt eine maschinenlesbare "
        "Kennzeichnung synthetischer Audioinhalte — verwenden Sie .wav "
        "oder .mp3."
    ),
    "error.marking_opt_out_needs_attestation": (
        "Das Deaktivieren von Wasserzeichen, C2PA-Signatur oder gesprochenem "
        "Hinweis verringert die Herkunftskennzeichnung nach EU AI Act "
        "Art. 50. Bestätigen Sie mit „Kennzeichnungsverantwortung "
        "übernehmen“, dass die Pflicht zur Kennzeichnung und Offenlegung "
        "dieser Ausgabe bei Ihnen als Betreiber liegt."
    ),
    "status.tts_initializing": "TTS-Backend wird initialisiert: {backend}",
    "status.tts_synthesizing": "Synthese mit {backend} …",
    "status.audio_saved": "Audio gespeichert unter: {path}",
    "warn.speaker_db_consent": (
        "Die Sprecherregistrierung speichert biometrische Stimmdaten mit "
        "Personenbezug. Stellen Sie sicher, dass eine Rechtsgrundlage und "
        "die Einwilligung der Person vorliegen (DSGVO Art. 9); bestätigen "
        "Sie dies mit --speaker-db-consent."
    ),
    "label.speaker_identity": "Voreingestellte Stimme ist:",
    "opt.speaker_identity_default": "(mitgelieferte Einstufung)",
    "opt.speaker_identity_real": "eine reale Person",
    "opt.speaker_identity_synthetic": "synthetisch",
    "opt.speaker_identity_unknown": "unbekannt",
    "tip.speaker_identity": (
        "Ob die voreingestellte Stimme dieses Backends einer identifizierbaren "
        "Person gehört. Wenn ja, ist die Ausgabe ein Deepfake nach Art. 3(60) "
        "der KI-Verordnung, und ein hörbarer Hinweis wird auch ohne "
        "Stimmklonen vorangestellt. Belassen Sie die mitgelieferte Einstufung, "
        "sofern Sie es nicht besser wissen."
    ),
    "disclosure.spoken": ("Die folgende Aufnahme wurde von künstlicher Intelligenz erzeugt."),
    "notice.synthetic_text": (
        "Dieser Text wurde von einem maschinellen Übersetzungsmodell erzeugt. "
        "Er verliert Nuancen und kann die Bedeutung umkehren — lassen Sie ihn "
        "von einem Menschen prüfen. Wenn Sie ihn veröffentlichen, um die "
        "Öffentlichkeit über Angelegenheiten von öffentlichem Interesse zu "
        "informieren, obliegt Ihnen die Offenlegung, dass er KI-generiert ist "
        "(KI-Verordnung Art. 50(4))."
    ),
    # --- KI-Kompetenz (Art. 4) / Zweckbestimmung ---------------------------
    "action.ai_notice": "Über KI in Susurrus",
    "msg.about.title": "Über Susurrus",
    # --- Ausführliche Hilfe und Status --------------------------------------
    "help.hf_token": (
        "Für die Sprechertrennung wird ein Hugging-Face-API-Token benötigt.\n\n"
        "1. Kostenloses Konto auf https://huggingface.co anlegen\n"
        "2. https://huggingface.co/settings/tokens öffnen\n"
        "3. Neues Token mit Leserechten („read“) erstellen\n"
        "4. Token hier einfügen\n\n"
        "Hinweis: Die Nutzungsbedingungen der Diarisierungsmodelle müssen "
        "unter https://huggingface.co/pyannote/speaker-diarization "
        "akzeptiert werden."
    ),
    "help.diarization_model": (
        "Wählen Sie das passende Diarisierungsmodell für Ihre Aufnahme:\n\n"
        "• Standard: Allgemeines Diarisierungsmodell\n"
        "• Englisch: Optimiert für englische Gespräche\n"
        "• Chinesisch: Optimiert für Mandarin-Gespräche\n"
        "• Deutsch: Optimiert für deutsche Gespräche\n"
        "• Spanisch: Optimiert für spanische Gespräche\n"
        "• Japanisch: Optimiert für japanische Gespräche\n\n"
        "Sprachspezifische Modelle liefern für die jeweilige Sprache oft "
        "bessere Ergebnisse, besonders bei Telefonaten und natürlichen "
        "Gesprächen."
    ),
    "help.mistral_key": (
        "<h3>Mistral-AI-API-Schlüssel</h3>"
        "<p>Der Mistral-API-Schlüssel wird für das Backend "
        "<b>voxtral-api</b> benötigt.</p>"
        "<h4>So erhalten Sie Ihren Schlüssel:</h4>"
        "<ol>"
        "<li>Kostenloses Konto auf "
        "<a href='https://console.mistral.ai/'>console.mistral.ai</a> anlegen</li>"
        "<li>Zum Bereich „API Keys“ wechseln</li>"
        "<li>Neuen API-Schlüssel erstellen</li>"
        "<li>Schlüssel hier einfügen</li>"
        "</ol>"
        "<h4>Alternativ:</h4>"
        "<p>Sie können die Umgebungsvariable <code>MISTRAL_API_KEY</code> "
        "setzen:</p>"
        "<p><b>PowerShell:</b> <code>$env:MISTRAL_API_KEY = 'ihr-schlüssel'</code></p>"
        "<p><b>CMD:</b> <code>set MISTRAL_API_KEY=ihr-schlüssel</code></p>"
        "<p><b>Linux/Mac:</b> <code>export MISTRAL_API_KEY='ihr-schlüssel'</code></p>"
    ),
    "help.diarization_unavailable": (
        "Sprechertrennung ist nicht verfügbar. Bitte stellen Sie sicher:\n\n"
        "1. pyannote.audio ist installiert\n"
        "2. Ein gültiges Hugging-Face-Token liegt in der Umgebungsvariablen "
        "HF_TOKEN\n\n"
        "Falls diese Meldung weiterhin erscheint, liegt möglicherweise ein "
        "Versionskonflikt zwischen Paketen vor."
    ),
    "label.history_entries": "{count} Einträge",
    "msg.delete_entry.body": "„{title}“ löschen?",
    "error.delete_failed": "Löschen fehlgeschlagen: {error}",
    "error.clear_failed": "Leeren fehlgeschlagen: {error}",
    "error.read_file": "Fehler beim Lesen der Datei: {error}",
    "error.generic": "Fehler: {error}",
    "error.watermark_detection_failed": "Wasserzeichenerkennung fehlgeschlagen: {error}",
    "error.save_failed": "Speichern fehlgeschlagen: {error}",
    "error.dependency_check": "Fehler bei der Abhängigkeitsprüfung: {error}",
    "msg.backend_unavailable.body": (
        "Das Backend „{backend}“ ist nicht in die CrispASR-Binärdatei "
        "einkompiliert.\n\n"
        "Verfügbare Backends: {available}\n\n"
        "Bauen Sie CrispASR mit dem benötigten Backend neu."
    ),
    "msg.play_failed.body": (
        "Audio konnte nicht abgespielt werden: {error}\n\n"
        "Die Datei ist gespeichert unter: {path}"
    ),
    "msg.saved.title": "Gespeichert",
    "msg.saved_to": "Gespeichert unter: {path}",
    "status.transcription_aborted": "Transkription vom Benutzer abgebrochen.",
    "status.starting_synthesis": "Synthese wird gestartet …",
    "status.server_stopped": "Server gestoppt.",
    "status.server_started": "Server gestartet auf {host}:{port}",
    "status.server_marking_proxy": (
        "Audio-Antworten werden als KI-generiert gekennzeichnet, bevor sie "
        "diesen Prozess verlassen (EU-KI-Verordnung Art. 50(2)). Antworten, "
        "die nicht gekennzeichnet werden können, werden abgelehnt statt "
        "ungekennzeichnet ausgeliefert."
    ),
    "msg.server_proxy_failed": (
        "Der KI-Kennzeichnungs-Proxy konnte nicht gestartet werden: {reason}.\n\n"
        "Der Server wurde nicht gestartet. Ohne den Proxy ist Susurrus nicht im "
        "Antwortpfad und kann die synthetischen Audiodaten des Endpunkts weder "
        "kennzeichnen noch prüfen, was Art. 50(2) der KI-Verordnung verlangt."
    ),
    "status.tts_done": "Fertig! Audio gespeichert unter: {path}",
    "status.install_issues": "Installation mit Problemen abgeschlossen: {detail}",
    "msg.install_voxtral.body": (
        "Dies installiert die Entwicklungsversion von transformers.\n\n"
        "Folgende Pakete werden installiert:\n"
        "• transformers (von GitHub)\n"
        "• mistral-common[audio]\n"
        "• soundfile\n\n"
        "Dies kann einige Minuten dauern. Fortfahren?"
    ),
    "msg.diarization_info.body": (
        "Sie verwenden die Sprechertrennung zum ersten Mal.\n\n"
        "Wichtige Hinweise:\n"
        "- Beim ersten Lauf wird das Diarisierungsmodell geladen (ca. 1 GB)\n"
        "- Die Verarbeitung kann länger dauern als eine normale Transkription\n"
        "- Für anderssprachige Inhalte empfiehlt sich das passende "
        "Sprachmodell\n\n"
        "Möchten Sie fortfahren?"
    ),
    "help.ffmpeg_install": (
        "1. Laden Sie den „essentials“-Build herunter\n"
        "2. Entpacken Sie das ZIP-Archiv nach C:\\ffmpeg\n"
        "3. Fügen Sie C:\\ffmpeg\\bin zum System-PATH hinzu:\n"
        "   - Systemsteuerung > System > Erweiterte Systemeinstellungen\n"
        "   - Auf „Umgebungsvariablen“ klicken\n"
        "   - Variable „Path“ bearbeiten und C:\\ffmpeg\\bin ergänzen\n"
        "   - Mit OK bestätigen und das Terminal neu starten"
    ),
    "help.voxtral_install": (
        "<h3>Voxtral-Abhängigkeiten installieren</h3>"
        "<p>Führen Sie diese Befehle aus:</p>"
        "<p><b>Windows:</b></p>"
        "<pre>install_voxtral.bat</pre>"
        "<p><b>Linux/Mac:</b></p>"
        "<pre>./install_voxtral.sh</pre>"
        "<p><b>Oder manuell:</b></p>"
        "<pre>pip uninstall transformers -y\n"
        "pip install git+https://github.com/huggingface/transformers.git\n"
        "pip install mistral-common[audio] soundfile</pre>"
    ),
    "help.diarization": (
        "<h2>Sprechertrennung in Susurrus</h2>"
        "<p>Die Sprechertrennung erkennt verschiedene Sprecher in Ihren "
        "Aufnahmen und erzeugt Transkripte mit Sprecherkennzeichnung.</p>"
        "<h3>Verfahren</h3>"
        "<ul>"
        "<li><b>PyAnnote</b> — neuronales Modell (HF-Token erforderlich)</li>"
        "<li><b>CrispASR-Verfahren</b> — energy, xcorr, vad-turns, sherpa, ecapa</li>"
        "</ul>"
        "<h3>Voraussetzungen (PyAnnote)</h3>"
        "<ul>"
        "<li>Hugging-Face-Konto und API-Token</li>"
        "<li>Installierte Bibliothek pyannote.audio</li>"
        "<li>Zustimmung zu den Modelllizenzen</li>"
        "</ul>"
        "<h3>Tipps für gute Ergebnisse</h3>"
        "<ul>"
        "<li>Saubere Aufnahmen mit wenig Störgeräuschen verwenden</li>"
        "<li>Für nicht-englische Inhalte sprachspezifische Modelle wählen</li>"
        "<li>Min./max. Sprecherzahl angeben, wenn sie bekannt ist</li>"
        "</ul>"
    ),
    "msg.about.body": (
        "<h1>{name}</h1>"
        "<p>Suite für Transkription, Sprachsynthese, Übersetzung und S2S</p>"
        "<p>Version {version}</p>"
        "<p>Funktionen:</p>"
        "<ul>"
        "<li>38+ ASR-Backends über CrispASR</li>"
        "<li>27+ TTS-Engines (lokal und Cloud)</li>"
        "<li>Mehrsprachige Übersetzung (m2m100, MadLad, Gemma4)</li>"
        "<li>Sprache-zu-Sprache (lfm2-audio, mini-omni2)</li>"
        "<li>Sprechertrennung (PyAnnote + CrispASR-Verfahren)</li>"
        "<li>Export: SRT, VTT, JSON, CSV, TXT</li>"
        "<li>Transkriptionsverlauf mit Suche</li>"
        "<li>Stapelverarbeitung</li>"
        "<li>Helles und dunkles Design</li>"
        "<li>Wellenformanzeige, Fortschritt in Echtzeit</li>"
        "<li>Eigenständige Alignierung (--align-only)</li>"
        "</ul>"
        "<p>Tastenkürzel: F5=Transkribieren, Strg+S=Speichern, "
        "Strg+T=Design, Strg+H=Verlauf</p>"
        "<p>Synthetische Audioinhalte werden als KI-generiert gekennzeichnet "
        "— siehe <i>Hilfe &gt; Über KI in Susurrus</i>.</p>"
    ),
    "msg.ai_notice.title": "Über KI in Susurrus",
    "msg.ai_notice.body": (
        "<h3>Dies ist ein KI-System</h3>"
        "<p>Susurrus transkribiert, synthetisiert und übersetzt Sprache und "
        "trennt Sprecher mithilfe von Modellen des maschinellen Lernens. "
        "Alle Ergebnisse sind eine <b>Vorhersage des Modells, kein "
        "Protokoll</b>.</p>"
        "<h4>Zweckbestimmung</h4>"
        "<p>Ein lokal arbeitendes Werkzeug für eigenes oder einwilligungs"
        "basiertes Material, dessen Ausgabe vor der Verwendung von einem "
        "Menschen geprüft wird.</p>"
        "<h4>Bekannte Grenzen</h4>"
        "<ul>"
        "<li><b>Transkription</b> enthält Fehler. Die Genauigkeit hängt stark "
        "von Akzent, Aufnahmequalität, Störgeräuschen, Fachvokabular und "
        "Sprache ab; nicht-muttersprachliche Akzente und Sprachen mit wenig "
        "Trainingsmaterial schneiden meist schlechter ab.</li>"
        "<li><b>Sprechertrennung</b> schätzt Sprecherwechsel und -anzahl. "
        "Überlappende Rede, ähnliche Stimmen und kurze Beiträge scheitern "
        "häufig.</li>"
        "<li><b>Übersetzung</b> verliert Nuancen und kann die Bedeutung "
        "umkehren, besonders bei Verneinung, Redewendungen und mehrdeutigen "
        "Pronomen.</li>"
        "<li><b>Synthetische Sprache</b> ist kein Beleg dafür, dass eine "
        "reale Person etwas gesagt hat.</li>"
        "</ul>"
        "<h4>Nicht validiert für</h4>"
        "<p>Jede Verwendung, bei der ein Fehler ohne menschliche Prüfung "
        "rechtliche oder sicherheitsrelevante Folgen hat: Beweismittel"
        "protokolle, medizinische Dokumentation, Entscheidungen über "
        "Beschäftigung oder Bildung, Kredit- oder Leistungsansprüche, "
        "Strafverfolgung, Migration und Grenzkontrolle.</p>"
        "<h4>Ihre Pflichten</h4>"
        "<p>Synthetische Audioinhalte werden automatisch als KI-generiert "
        "gekennzeichnet. Die Offenlegung gegenüber den Personen, die die "
        "Ausgabe sehen oder hören, bleibt Ihre Aufgabe. Siehe COMPLIANCE.md.</p>"
    ),
    "warn.no_transcription": "Keine Transkription zum Speichern vorhanden.",
    # --- Wizard -------------------------------------------------------------
    "wizard.title": "Stimmklon-Assistent",
    "wizard.step1_title": "Schritt 1: Referenzaudio auswählen",
    "wizard.step1_body": ("Wählen Sie eine .wav-Datei mit der zu klonenden Stimme aus."),
    "wizard.step2_title": "Schritt 2: Referenztranskription",
    "wizard.step2_body": (
        "Geben Sie ein, was im Referenzaudio gesprochen wird. "
        "Das hilft der TTS-Engine, die Stimme nachzubilden."
    ),
    "wizard.step3_title": "Schritt 3: Bestätigen",
    "wizard.step3_body": (
        "Das Klonen von Stimmen erfordert eine Einwilligungsbestätigung " "(EU AI Act)."
    ),
    "wizard.select_audio_dialog": "Referenzaudio auswählen",
    "wizard.summary_audio": "Audio:",
    "wizard.summary_ref_text": "Referenztext:",
    # --- Dialog titles ------------------------------------------------------
    "msg.error.title": "Fehler",
    "msg.warning.title": "Warnung",
    "msg.success.title": "Erfolg",
    "msg.timeout.title": "Zeitüberschreitung",
    # --- Dialog messages ----------------------------------------------------
    "msg.no_audio_input.title": "Keine Audioeingabe",
    "msg.no_audio_input.body": ("Bitte geben Sie eine Audiodatei oder eine URL an."),
    "msg.backend_unavailable.title": "Backend nicht verfügbar",
    "msg.unsupported_url.title": "Nicht unterstützte URL",
    "msg.unsupported_url.body": (
        "Das Herunterladen von YouTube wird aufgrund der Nutzungsbedingungen "
        "nicht unterstützt.\n\nBitte verwenden Sie eine lokale Datei oder "
        "eine direkte URL zu einer Mediendatei außerhalb von YouTube."
    ),
    "msg.missing_api_key.title": "Fehlender API-Schlüssel",
    "msg.missing_api_key.body": (
        "Für das voxtral-api-Backend ist ein Mistral-AI-API-Schlüssel "
        "erforderlich.\n\nBitte geben Sie ihn im Bereich „Voxtral-API-"
        "Einstellungen“ ein oder setzen Sie die Umgebungsvariable "
        "MISTRAL_API_KEY."
    ),
    "msg.no_text.title": "Kein Text",
    "msg.no_text_tts.body": ("Bitte geben Sie Text ein oder laden Sie eine Textdatei."),
    "msg.no_text_translate.body": "Bitte geben Sie den zu übersetzenden Text ein.",
    "msg.tts_error.title": "TTS-Fehler",
    "msg.no_audio.title": "Kein Audio",
    "msg.no_audio.body": "Keine TTS-Ausgabe zum Abspielen vorhanden.",
    "msg.playback_error.title": "Wiedergabefehler",
    "msg.translation_error.title": "Übersetzungsfehler",
    "msg.translating": "Übersetzung läuft...",
    "msg.translation_complete": "Übersetzung abgeschlossen.",
    "msg.crispasr_missing.body": "CrispASR-Programm nicht gefunden.",
    "msg.crispasr_missing_watermark.body": (
        "CrispASR-Programm für die Wasserzeichenerkennung nicht gefunden."
    ),
    "msg.no_audio_selected.body": "Keine Audiodatei ausgewählt.",
    "msg.watermark_detection.title": "Wasserzeichenerkennung",
    "msg.watermark_timeout.body": ("Zeitüberschreitung bei der Wasserzeichenerkennung."),
    "msg.dependency_check_error.title": "Fehler bei der Abhängigkeitsprüfung",
    "msg.ffmpeg_missing.title": "FFMPEG nicht gefunden",
    "msg.ffmpeg_missing.body": (
        "FFMPEG ist nicht installiert oder nicht im PATH. Die Unterstützung "
        "für Audioformate ist eingeschränkt.\n\nSie können FFMPEG über "
        "Werkzeuge > Abhängigkeiten installieren einrichten."
    ),
    "msg.cuda_missing.title": "PyTorch-CUDA-Unterstützung fehlt",
    "msg.cuda_missing.body": (
        "Es wurde eine NVIDIA-GPU erkannt, aber PyTorch wurde ohne "
        "CUDA-Unterstützung installiert.\n\nMöchten Sie PyTorch jetzt mit "
        "CUDA-Unterstützung neu installieren?"
    ),
    "msg.voxtral_missing.title": "Voxtral-Abhängigkeiten fehlen",
    "msg.voxtral_missing.body": (
        "Das voxtral-local-Backend benötigt die Entwicklungsversion von "
        "transformers.\n\nMöchten Sie die Installationsanleitung ansehen?"
    ),
    "msg.diarization_unavailable.title": "Sprechertrennung nicht verfügbar",
    "msg.missing_token.title": "Fehlender Token",
    "msg.missing_token.body": (
        "Für die Sprechertrennung ist ein Hugging-Face-Token erforderlich.\n\n"
        "Bitte geben Sie Ihren Token ein oder deaktivieren Sie die "
        "Sprechertrennung."
    ),
    "msg.diarization_info.title": "Hinweise zur Sprechertrennung",
    "msg.delete_entry.title": "Eintrag löschen",
    "msg.clear_history.title": "Verlauf löschen",
    "msg.clear_history.body": (
        "ALLE Verlaufseinträge löschen? Dies kann nicht rückgängig gemacht " "werden."
    ),
    "msg.select_ref_audio.body": "Bitte wählen Sie eine Referenzaudiodatei aus.",
    "msg.install_deps.title": "Abhängigkeiten installieren",
    "msg.install_complete": "Installation erfolgreich abgeschlossen!",
    "msg.preparing": "Wird vorbereitet...",
    "msg.install_pytorch_cuda.title": "PyTorch mit CUDA installieren",
    "msg.install_pytorch_cuda.body": (
        "PyTorch wird mit CUDA-Unterstützung neu installiert. Fortfahren?"
    ),
    "msg.install_voxtral.title": "Voxtral-Abhängigkeiten installieren",
    "msg.ffmpeg_instructions.title": "FFMPEG-Installationsanleitung",
    "msg.hf_token_help.title": "Hilfe zum Hugging-Face-Token",
    "msg.diarization_model_help.title": "Modellauswahl für die Sprechertrennung",
    "msg.mistral_key_help.title": "Hilfe zum Mistral-API-Schlüssel",
    "msg.voxtral_instructions.title": "Voxtral-Installationsanleitung",
    "msg.diarization_help.title": "Hilfe zur Sprechertrennung",
    "msg.language_changed.title": "Sprache geändert",
    "msg.language_changed.body": (
        "Die Sprache der Benutzeroberfläche wurde geändert. Starten Sie "
        "Susurrus neu, damit sie überall übernommen wird."
    ),
    "msg.audit_log.title": "Biometrisches Prüfprotokoll",
    "msg.audit_log_empty": "Es wurden keine biometrischen Ereignisse aufgezeichnet.",
    "msg.audit_log_valid": (
        "Prüfkette verifiziert: {count} Ereignisse, keine Manipulation erkannt."
    ),
    "msg.audit_log_invalid": "Prüfkette UNTERBROCHEN: {errors}",
    # --- Dialog headings ----------------------------------------------------
    "heading.dependency_status": "Status der Abhängigkeiten",
    "heading.dependencies_required": (
        "Die folgenden Abhängigkeiten werden von Susurrus benötigt oder " "empfohlen:"
    ),
    "heading.install_instructions": "Installationsanleitung:",
    "help.install_pip": (
        "Fehlende erforderliche Abhängigkeiten lassen sich mit pip "
        "installieren:<br>"
        "<code>pip install torch transformers pydub numpy</code><br><br>"
        "Für die Sprechertrennung zusätzlich:<br>"
        "<code>pip install pyannote.audio huggingface_hub</code><br><br>"
        "Für ffmpeg siehe "
        "<a href='https://ffmpeg.org/download.html'>ffmpeg.org/download.html</a>"
    ),
    "heading.install_missing": "Fehlende Abhängigkeiten installieren",
    "heading.deps_needed": (
        "Für den vollen Funktionsumfang werden folgende Abhängigkeiten " "benötigt:"
    ),
    "heading.cuda_diagnostics": "CUDA- und NVIDIA-Diagnose",
    "dialog.dependencies.title": "Susurrus-Abhängigkeiten",
    # --- Voxtral panel ------------------------------------------------------
    "voxtral.description": (
        "Voxtral ist das Spracherkennungsmodell von Mistral AI.\n"
        "Es unterstützt 8 Sprachen und bietet lokale sowie API-basierte "
        "Verarbeitung."
    ),
    "voxtral.languages": (
        "<b>Unterstützte Sprachen:</b> Englisch, Französisch, Spanisch, "
        "Deutsch, Italienisch, Portugiesisch, Polnisch, Niederländisch"
    ),
    "voxtral.note": (
        "<b>Hinweis:</b> voxtral-local benötigt transformers von GitHub. "
        "Führen Sie install_voxtral.sh/bat zur Einrichtung aus."
    ),
}
