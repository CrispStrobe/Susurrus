# gui/widgets/crispasr_advanced_settings.py
"""CrispASR advanced settings widget."""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from config import (
    CRISPASR_DIARIZE_METHODS,
    CRISPASR_LID_BACKENDS,
    CRISPASR_SUB_BACKENDS,
)

from .collapsible_box import CollapsibleBox


class CrispASRAdvancedSettingsBox(CollapsibleBox):
    """Advanced settings specific to the CrispASR backend.

    Provides controls for VAD, diarization, LID, alignment,
    speaker management, grammar, and streaming parameters.
    """

    def __init__(self, parent=None):
        super().__init__("CrispASR Advanced Options", parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # --- Sub-backend override ---
        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Sub-backend:"))
        self.sub_backend = QComboBox()
        self.sub_backend.setEditable(True)
        # Sourced from config so it stays in sync with the CrispASR release
        self.sub_backend.addItems(["(auto-detect)"] + list(CRISPASR_SUB_BACKENDS))
        sub_row.addWidget(self.sub_backend)
        layout.addLayout(sub_row)

        # --- VAD settings ---
        vad_row = QHBoxLayout()
        self.vad_enabled = QCheckBox("Enable VAD")
        vad_row.addWidget(self.vad_enabled)

        vad_row.addWidget(QLabel("VAD model:"))
        self.vad_model = QComboBox()
        self.vad_model.setEditable(True)
        self.vad_model.addItems(["(default)", "silero", "firered"])
        vad_row.addWidget(self.vad_model)

        vad_row.addWidget(QLabel("Threshold:"))
        self.vad_threshold = QLineEdit()
        self.vad_threshold.setPlaceholderText("0.5")
        self.vad_threshold.setMaximumWidth(60)
        vad_row.addWidget(self.vad_threshold)
        layout.addLayout(vad_row)

        # --- Inference settings ---
        inf_row = QHBoxLayout()
        inf_row.addWidget(QLabel("Temperature:"))
        self.temperature = QLineEdit()
        self.temperature.setPlaceholderText("0.0")
        self.temperature.setMaximumWidth(60)
        inf_row.addWidget(self.temperature)

        inf_row.addWidget(QLabel("Beam size:"))
        self.beam_size = QLineEdit()
        self.beam_size.setPlaceholderText("5")
        self.beam_size.setMaximumWidth(60)
        inf_row.addWidget(self.beam_size)

        inf_row.addWidget(QLabel("Best of:"))
        self.best_of = QLineEdit()
        self.best_of.setPlaceholderText("1")
        self.best_of.setMaximumWidth(60)
        inf_row.addWidget(self.best_of)

        inf_row.addWidget(QLabel("Seed:"))
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("0")
        self.seed.setMaximumWidth(80)
        inf_row.addWidget(self.seed)
        layout.addLayout(inf_row)

        # --- Diarization settings ---
        dia_row = QHBoxLayout()
        self.diarize_enabled = QCheckBox("Diarize")
        dia_row.addWidget(self.diarize_enabled)

        dia_row.addWidget(QLabel("Method:"))
        self.diarize_method = QComboBox()
        self.diarize_method.addItems(list(CRISPASR_DIARIZE_METHODS))
        dia_row.addWidget(self.diarize_method)

        dia_row.addWidget(QLabel("Max speakers:"))
        self.diarize_max_speakers = QLineEdit()
        self.diarize_max_speakers.setPlaceholderText("0")
        self.diarize_max_speakers.setMaximumWidth(60)
        dia_row.addWidget(self.diarize_max_speakers)
        layout.addLayout(dia_row)

        # --- LID settings ---
        lid_row = QHBoxLayout()
        self.detect_language = QCheckBox("Detect language")
        lid_row.addWidget(self.detect_language)

        lid_row.addWidget(QLabel("LID backend:"))
        self.lid_backend = QComboBox()
        self.lid_backend.addItems(list(CRISPASR_LID_BACKENDS))
        lid_row.addWidget(self.lid_backend)
        layout.addLayout(lid_row)

        # --- Misc flags ---
        misc_row = QHBoxLayout()
        self.split_on_punct = QCheckBox("Split on punct")
        misc_row.addWidget(self.split_on_punct)

        self.flash_attn = QCheckBox("Flash attention")
        misc_row.addWidget(self.flash_attn)

        self.auto_download = QCheckBox("Auto-download")
        self.auto_download.setChecked(True)
        misc_row.addWidget(self.auto_download)

        self.translate_to_en = QCheckBox("Translate to EN")
        misc_row.addWidget(self.translate_to_en)
        layout.addLayout(misc_row)

        # --- Punctuation & alignment ---
        pa_row = QHBoxLayout()
        pa_row.addWidget(QLabel("Punc model:"))
        self.punc_model = QComboBox()
        self.punc_model.setEditable(True)
        self.punc_model.addItems(["(none)", "auto", "firered"])
        pa_row.addWidget(self.punc_model)

        pa_row.addWidget(QLabel("Aligner:"))
        self.aligner_model = QLineEdit()
        self.aligner_model.setPlaceholderText("(none)")
        pa_row.addWidget(self.aligner_model)
        layout.addLayout(pa_row)

        # --- Hotwords (contextual biasing) ---
        hot_row = QHBoxLayout()
        hot_row.addWidget(QLabel("Hotwords:"))
        self.hotwords = QLineEdit()
        self.hotwords.setPlaceholderText(
            "comma-separated, e.g. Tokyo,CrispASR (parakeet/qwen3/voxtral)"
        )
        hot_row.addWidget(self.hotwords)

        hot_row.addWidget(QLabel("Boost:"))
        self.hotwords_boost = QLineEdit()
        self.hotwords_boost.setPlaceholderText("2.0")
        self.hotwords_boost.setMaximumWidth(60)
        hot_row.addWidget(self.hotwords_boost)
        layout.addLayout(hot_row)

        # --- Prompt ---
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Initial prompt:"))
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Optional initial prompt for decoding")
        prompt_row.addWidget(self.prompt)
        layout.addLayout(prompt_row)

        self.setContentLayout(layout)

    def get_kwargs(self):
        """Collect all settings as a kwargs dict for CrispasrBackend."""
        kwargs = {}

        sub = self.sub_backend.currentText()
        if sub and sub != "(auto-detect)":
            kwargs["crispasr_backend"] = sub

        if self.vad_enabled.isChecked():
            kwargs["vad"] = True
        vad_model = self.vad_model.currentText()
        if vad_model and vad_model != "(default)":
            kwargs["vad_model"] = vad_model
        vt = self.vad_threshold.text().strip()
        if vt:
            kwargs["vad_threshold"] = float(vt)

        temp = self.temperature.text().strip()
        if temp:
            kwargs["temperature"] = float(temp)
        bs = self.beam_size.text().strip()
        if bs:
            kwargs["beam_size"] = int(bs)
        bo = self.best_of.text().strip()
        if bo:
            kwargs["best_of"] = int(bo)
        seed = self.seed.text().strip()
        if seed:
            kwargs["seed"] = int(seed)

        if self.diarize_enabled.isChecked():
            kwargs["diarize"] = True
            kwargs["diarize_method"] = self.diarize_method.currentText()
            ms = self.diarize_max_speakers.text().strip()
            if ms:
                kwargs["diarize_max_speakers"] = int(ms)

        if self.detect_language.isChecked():
            kwargs["detect_language"] = True
            kwargs["lid_backend"] = self.lid_backend.currentText()

        if self.split_on_punct.isChecked():
            kwargs["split_on_punct"] = True
        if self.flash_attn.isChecked():
            kwargs["flash_attn"] = True
        if self.auto_download.isChecked():
            kwargs["auto_download"] = True
        if self.translate_to_en.isChecked():
            kwargs["translate"] = True

        punc = self.punc_model.currentText()
        if punc and punc != "(none)":
            kwargs["punc_model"] = punc
        aligner = self.aligner_model.text().strip()
        if aligner and aligner != "(none)":
            kwargs["aligner_model"] = aligner

        hotwords = self.hotwords.text().strip()
        if hotwords:
            kwargs["hotwords"] = hotwords
            boost = self.hotwords_boost.text().strip()
            if boost:
                kwargs["hotwords_boost"] = float(boost)

        prompt = self.prompt.text().strip()
        if prompt:
            kwargs["prompt"] = prompt

        return kwargs
