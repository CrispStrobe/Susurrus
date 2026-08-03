"""Read a few string keys out of a GGUF file's metadata block.

Susurrus classifies whose voice a preset is (:mod:`utils.speaker_identity`) and
had to do it by matching the checkpoint's *file name*, against this project's
own rule about not trusting names. That was accepted only because the failure
is safe and the alternative was no answer at all.

CrispASR has since stamped the answer into the checkpoint itself —
``crispasr.voice.speaker_identity`` plus a companion ``…_evidence`` — so the
authoritative answer travels with the weights and survives a rename. This reads
it.

Deliberately not a GGUF library. It parses the header far enough to walk the
key/value block, returns the handful of strings asked for, and gives up on
anything surprising: an unreadable or unexpected file yields ``{}``, which the
caller reads as "no stamp" and falls back to what it did before. A metadata
reader that raised would turn a provenance *improvement* into a synthesis
failure, which is the wrong trade for a hint.
"""

import logging
import struct

logger = logging.getLogger(__name__)

_MAGIC = b"GGUF"

#: How much of the file to consider. The KV block sits at the front, ahead of
#: the tensor data; a checkpoint can be gigabytes and none of that is metadata.
_MAX_HEADER = 8 << 20

# GGUF value type tags.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

#: Fixed-width types, so a value we do not care about can be stepped over
#: rather than decoded.
_FIXED_WIDTH = {
    _UINT8: 1,
    _INT8: 1,
    _BOOL: 1,
    _UINT16: 2,
    _INT16: 2,
    _UINT32: 4,
    _INT32: 4,
    _FLOAT32: 4,
    _UINT64: 8,
    _INT64: 8,
    _FLOAT64: 8,
}


class _Cursor:
    """A bounds-checked reader over the header bytes."""

    def __init__(self, data):
        self._data = data
        self._at = 0

    def take(self, count):
        if count < 0 or self._at + count > len(self._data):
            raise ValueError("truncated GGUF header")
        chunk = self._data[self._at : self._at + count]
        self._at += count
        return chunk

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        length = self.u64()
        if length > _MAX_HEADER:
            raise ValueError("implausible string length in GGUF header")
        return self.take(length).decode("utf-8", errors="replace")

    def skip_value(self, value_type):
        """Step over a value of *value_type* without decoding it."""
        if value_type in _FIXED_WIDTH:
            self.take(_FIXED_WIDTH[value_type])
        elif value_type == _STRING:
            self.string()
        elif value_type == _ARRAY:
            element_type = self.u32()
            count = self.u64()
            if element_type in _FIXED_WIDTH:
                # Multiplied inside take(), which bounds-checks the result, so
                # an absurd count fails cleanly rather than allocating.
                self.take(_FIXED_WIDTH[element_type] * count)
            elif element_type == _STRING:
                for _ in range(min(count, _MAX_HEADER)):
                    self.string()
            else:
                raise ValueError(f"unsupported GGUF array element type {element_type}")
        else:
            raise ValueError(f"unsupported GGUF value type {value_type}")


def read_string_keys(path, keys):
    """Return ``{key: value}`` for the *keys* present as GGUF strings in *path*.

    Missing keys are simply absent from the result. Anything that is not a
    readable GGUF file — including a plain WAV, a directory, or a truncated
    download — yields an empty dict rather than an error.
    """
    wanted = set(keys or ())
    if not path or not wanted:
        return {}

    try:
        with open(path, "rb") as handle:
            data = handle.read(_MAX_HEADER)
    except OSError:
        return {}

    if not data.startswith(_MAGIC):
        return {}

    found = {}
    try:
        cursor = _Cursor(data)
        cursor.take(4)  # magic
        cursor.u32()  # format version
        cursor.u64()  # tensor count
        kv_count = cursor.u64()

        for _ in range(kv_count):
            key = cursor.string()
            value_type = cursor.u32()
            if key in wanted and value_type == _STRING:
                found[key] = cursor.string()
                if len(found) == len(wanted):
                    break
            else:
                cursor.skip_value(value_type)
    except (ValueError, struct.error, UnicodeDecodeError) as e:
        # A partially parsed header still yields whatever was read before the
        # problem, which is strictly better than discarding a valid stamp
        # because some later value used a type this reader does not know.
        logger.debug("Stopped reading GGUF metadata from %s: %s", path, e)

    return found
