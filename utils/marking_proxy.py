"""Marking reverse proxy for server mode (EU AI Act Art. 50(2)).

``--mode server`` used to hand the socket straight to the CrispASR binary.
Susurrus was not in the response path, so it could neither verify what the
binary marked nor apply the declarative floor it applies on every other route —
and an HTTP endpoint is precisely where synthetic audio reaches people who
never see a warning printed on someone else's terminal.

This module puts Susurrus back in the path. It binds the port the operator
asked for, starts the binary on loopback, and forwards between them:

* **Audio responses are marked.** ``audio/*`` bodies are buffered, run through
  the same :func:`utils.provenance.apply_provenance` pipeline as a local
  synthesis, and forwarded with the marked bytes. If nothing can be applied,
  the response is replaced with a 502 rather than served unmarked — the gate
  works the same way here as it does on disk.
* **Everything else is passed through untouched**, and streamed rather than
  buffered. Transcription JSON, chat completions and Server-Sent Events must
  not be held back: a proxy that buffers an SSE stream turns a working
  endpoint into one that appears to hang.

What it deliberately does not do: protocol upgrades. A WebSocket handshake
cannot be forwarded by this proxy, so it is refused with a 501 naming why,
rather than half-forwarded into a connection that dies later for reasons
nobody can trace.

The proxy is stdlib-only. Requiring an optional package for the layer whose
entire purpose is to work without optional packages would defeat it.
"""

import logging
import os
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

#: Headers that describe a single hop and must not be forwarded.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

#: Content types we can name a container for, so marking has an extension to
#: dispatch on. Anything else audio is still refused rather than passed.
_AUDIO_EXTENSIONS = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
}

_READ_CHUNK = 64 * 1024


def find_free_port(host="127.0.0.1"):
    """Return a port that is free right now on *host*.

    Inherently racy — something else can take it between this call and the
    bind. The alternative is a fixed port that collides deterministically, so
    the race is the better failure mode: it is rare, and it surfaces as a bind
    error at startup rather than as a silent misroute.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_upstream(host, port, timeout=60.0, interval=0.2):
    """Block until *host:port* accepts a connection. True if it came up."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(interval)
    return False


def _content_type(headers):
    return (headers.get("Content-Type") or "").split(";")[0].strip().lower()


def _audio_extension(content_type):
    """Return the file extension for an audio content type, or None."""
    if content_type in _AUDIO_EXTENSIONS:
        return _AUDIO_EXTENSIONS[content_type]
    if content_type.startswith("audio/"):
        # An audio type we do not have a container name for. Returning None
        # here would classify it as "not audio" and pass it through unmarked,
        # which is the one outcome this proxy exists to prevent.
        return ""
    return None


class _LimitedReader:
    """Read at most *length* bytes from *stream*, for streaming uploads.

    ``http.client`` reads file-like bodies in blocks, so handing it one of
    these forwards a large transcription upload without holding it in memory.
    """

    def __init__(self, stream, length):
        self._stream = stream
        self._remaining = length

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self._remaining
        data = self._stream.read(min(size, self._remaining))
        self._remaining -= len(data)
        return data


def _read_chunked(stream):
    """De-chunk a ``Transfer-Encoding: chunked`` request body.

    BaseHTTPRequestHandler does not do this. Refusing chunked uploads would be
    simpler, but it would reject a legitimate client for the proxy's
    convenience, so it is handled — buffered, because chunked requests to these
    endpoints are small and rare.
    """
    body = b""
    while True:
        line = stream.readline(65536).strip()
        if not line:
            break
        try:
            size = int(line.split(b";")[0], 16)
        except ValueError:
            break
        if size == 0:
            # Consume trailers up to the terminating blank line.
            while True:
                trailer = stream.readline(65536)
                if not trailer or trailer in (b"\r\n", b"\n"):
                    break
            break
        body += stream.read(size)
        stream.read(2)  # trailing CRLF
    return body


class MarkingProxy:
    """A reverse proxy that marks synthetic audio on its way to the client."""

    def __init__(
        self,
        listen_host,
        listen_port,
        upstream_host,
        upstream_port,
        options=None,
        model=None,
    ):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.options = options or {}
        self.model = model
        self._server = None
        self._thread = None
        #: Counters, so an operator can tell whether marking is actually
        #: happening rather than assuming it from the absence of errors.
        self.marked_responses = 0
        self.refused_responses = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Bind and serve in a background thread. Raises OSError if it cannot."""
        proxy = self

        class Handler(_ProxyHandler):
            parent = proxy

        self._server = ThreadingHTTPServer((self.listen_host, self.listen_port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            "Marking proxy listening on %s:%d -> %s:%d",
            self.listen_host,
            self.listen_port,
            self.upstream_host,
            self.upstream_port,
        )
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- marking -----------------------------------------------------------

    def mark_audio(self, body, extension):
        """Return marked *body*, or None if no layer could be applied.

        None means the response must not be served. The caller turns that into
        a 502: the alternative is forwarding unmarked synthetic audio, which is
        exactly what this proxy was added to stop.
        """
        from utils.provenance import ProvenanceError, complete_marking

        if not extension:
            return None

        fd, path = tempfile.mkstemp(suffix=extension, prefix="susurrus-proxy-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(body)
            try:
                # complete_marking, not apply_provenance: this audio came from
                # the binary, which marks what it can. Applying every layer
                # regardless stacked a second watermark on the first — measured
                # at ~41 dB SNR cost on real Piper output whose mark was
                # already above threshold — and would stack a second C2PA
                # manifest too.
                complete_marking(path, options=self.options, model=self.model)
            except ProvenanceError as e:
                # complete_marking has already deleted the file.
                logger.warning("Refusing to serve unmarked synthetic audio: %s", e)
                self.refused_responses += 1
                return None
            with open(path, "rb") as f:
                marked = f.read()
        except OSError as e:
            logger.warning("Could not mark proxied audio: %s", e)
            self.refused_responses += 1
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        self.marked_responses += 1
        return marked


class _ProxyHandler(BaseHTTPRequestHandler):
    """Forward one request upstream and mark the response if it is audio."""

    parent = None
    protocol_version = "HTTP/1.1"
    server_version = "Susurrus-MarkingProxy"

    def log_message(self, fmt, *args):  # noqa: A003 - BaseHTTPRequestHandler API
        logger.debug("proxy: " + fmt, *args)

    # Every method routes through the same path; the proxy does not care which
    # verb it is forwarding.
    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_PATCH(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    # -- internals ---------------------------------------------------------

    def _refuse(self, status, message):
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass
        self.close_connection = True

    def _request_body(self):
        """Return the request body as bytes or a streaming file-like object."""
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            return _read_chunked(self.rfile), None
        length = self.headers.get("Content-Length")
        if not length:
            return None, None
        try:
            length = int(length)
        except ValueError:
            return None, None
        return _LimitedReader(self.rfile, length), length

    def _forward_headers(self):
        headers = {}
        for key, value in self.headers.items():
            if key.lower() in _HOP_BY_HOP or key.lower() == "content-length":
                continue
            headers[key] = value
        headers["Host"] = f"{self.parent.upstream_host}:{self.parent.upstream_port}"
        return headers

    def _proxy(self):
        import http.client

        # A protocol upgrade cannot survive this proxy: http.client gives us a
        # parsed response, not the raw socket, so a 101 would leave both sides
        # believing a tunnel exists. Say so instead of failing later.
        if (self.headers.get("Upgrade") or "").strip():
            self._refuse(
                501,
                '{"error":"Susurrus\'s EU AI Act marking proxy cannot forward '
                "protocol upgrades (WebSocket). Marking is applied to audio "
                "responses, and an upgraded connection would bypass it. Use the "
                "HTTP endpoints, or run the server with "
                '--accept-marking-responsibility to bypass the proxy."}',
            )
            return

        body, length = self._request_body()
        headers = self._forward_headers()
        if length is not None:
            headers["Content-Length"] = str(length)

        try:
            conn = http.client.HTTPConnection(
                self.parent.upstream_host, self.parent.upstream_port, timeout=600
            )
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
        except OSError as e:
            logger.warning("Upstream connection failed: %s", e)
            self._refuse(502, '{"error":"upstream CrispASR server is unreachable"}')
            return

        try:
            self._relay(response)
        finally:
            conn.close()

    def _relay(self, response):
        content_type = _content_type(response.headers)
        extension = _audio_extension(content_type)

        if extension is not None and self.command != "HEAD":
            self._relay_audio(response, extension)
        else:
            self._relay_passthrough(response)

    def _relay_audio(self, response, extension):
        """Buffer, mark, forward. Audio is the one thing worth holding."""
        body = response.read()
        marked = self.parent.mark_audio(body, extension)
        if marked is None:
            self._refuse(
                502,
                '{"error":"Susurrus refused to serve synthetic audio it could '
                "not mark. EU AI Act Art. 50(2) requires machine-readable "
                "marking of synthetic audio. Serve .wav or .mp3, install "
                "'susurrus[tts]', or run the server with "
                '--accept-marking-responsibility."}',
            )
            return

        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() in _HOP_BY_HOP or key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(marked)))
        # Names the marking in-band, so a client can tell the difference
        # between "marked by Susurrus" and "whatever the binary happened to do".
        self.send_header("X-Susurrus-AI-Marked", "EU-AI-Act-Art50-2")
        self.end_headers()
        try:
            self.wfile.write(marked)
        except OSError:
            pass

    def _relay_passthrough(self, response):
        """Stream a non-audio response through without holding it.

        Buffering here would break Server-Sent Events and streaming
        transcription: the client would see nothing until the stream ended,
        which for a chat completion is indistinguishable from a hang.
        """
        content_length = response.headers.get("Content-Length")

        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() in _HOP_BY_HOP:
                continue
            self.send_header(key, value)
        if content_length is None:
            # Length unknown (chunked or SSE upstream). Signal end-of-body by
            # closing, rather than re-chunking a stream we do not need to
            # reshape.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

        if self.command == "HEAD":
            return

        # read1() returns whatever has arrived, where read() blocks until it
        # has the full amount or the stream ends. With read() an SSE stream is
        # buffered to completion and the client sees nothing until the server
        # is done — measurably so: every event landed at once, at the time the
        # last one was produced. That is the exact regression this branch of
        # the relay exists to avoid.
        read = getattr(response, "read1", None) or response.read

        try:
            while True:
                chunk = read(_READ_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()  # SSE clients need each event as it arrives
        except OSError:
            self.close_connection = True
