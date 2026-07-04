"""Segment model for transcription results — supports editing, speaker names, confidence."""


class Segment:
    """A single transcription segment with editing support."""

    __slots__ = ("start", "end", "text", "speaker", "confidence", "edited")

    def __init__(self, start=0.0, end=0.0, text="", speaker=None, confidence=None, edited=False):
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker
        self.confidence = confidence
        self.edited = edited

    def to_dict(self):
        d = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker:
            d["speaker"] = self.speaker
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.edited:
            d["edited"] = True
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            start=d.get("start", 0.0),
            end=d.get("end", 0.0),
            text=d.get("text", ""),
            speaker=d.get("speaker"),
            confidence=d.get("confidence"),
            edited=d.get("edited", False),
        )

    @classmethod
    def from_tuple(cls, t):
        return cls(start=t[0], end=t[1], text=t[2])

    def to_tuple(self):
        return (self.start, self.end, self.text)

    def __repr__(self):
        spk = f" [{self.speaker}]" if self.speaker else ""
        return f"Segment({self.start:.2f}-{self.end:.2f}{spk}: {self.text!r})"


class TranscriptionResult:
    """A collection of segments with speaker name mapping and edit tracking."""

    def __init__(self, segments=None):
        self.segments = segments or []
        self.speaker_names = {}  # "Speaker 1" → "Alice"

    def add_segment(self, start, end, text, speaker=None, confidence=None):
        seg = Segment(start=start, end=end, text=text, speaker=speaker, confidence=confidence)
        self.segments.append(seg)
        return seg

    def edit_segment(self, index, new_text):
        if 0 <= index < len(self.segments):
            self.segments[index].text = new_text
            self.segments[index].edited = True

    def rename_speaker(self, old_name, new_name):
        self.speaker_names[old_name] = new_name

    def display_speaker(self, speaker):
        if speaker is None:
            return None
        return self.speaker_names.get(speaker, speaker)

    def to_tuples(self):
        return [seg.to_tuple() for seg in self.segments]

    def full_text(self):
        return "\n".join(seg.text for seg in self.segments if seg.text)

    @classmethod
    def from_tuples(cls, tuples):
        result = cls()
        for t in tuples:
            result.segments.append(Segment.from_tuple(t))
        return result
