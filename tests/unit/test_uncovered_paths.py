"""Coverage for the three routes the audit left open.

Server mode served synthetic audio Susurrus never saw, the audit log could be
truncated from the tail undetectably, and translated text carried no disclosure
at all. Each is now closed as far as the software can close it; these tests pin
that, including the parts that remain limited.
"""

import io
import json
import os
import shutil
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from utils.marking_proxy import MarkingProxy, find_free_port, wait_for_upstream


def _wav_bytes(seconds=0.5, rate=24000):
    buf = io.BytesIO()
    n = int(seconds * rate)
    frames = struct.pack(f"<{n}h", *[(i % 200 - 100) * 80 for i in range(n)])
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


WAV = _wav_bytes()


class _Upstream(BaseHTTPRequestHandler):
    """Stands in for the CrispASR server: audio, JSON and an SSE stream."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/audio":
            self._send(200, "audio/wav", WAV)
        elif self.path == "/exotic":
            self._send(200, "audio/basic", b"\x00\x01\x02\x03")
        elif self.path == "/json":
            self._send(200, "application/json", json.dumps({"text": "hello"}).encode())
        elif self.path == "/sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                chunk = f"data: e{i}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                time.sleep(0.1)
            self.wfile.write(b"0\r\n\r\n")
        else:
            self._send(404, "text/plain", b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length)
        self._send(200, "application/json", json.dumps({"received": len(payload)}).encode())


class TestMarkingProxy(unittest.TestCase):
    """Server mode is in the response path now, and must stay there."""

    @classmethod
    def setUpClass(cls):
        cls.up_port = find_free_port()
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", cls.up_port), _Upstream)
        cls.upstream.daemon_threads = True
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        wait_for_upstream("127.0.0.1", cls.up_port, timeout=10)

        cls.proxy_port = find_free_port()
        cls.proxy = MarkingProxy(
            "127.0.0.1", cls.proxy_port, "127.0.0.1", cls.up_port, model="test"
        ).start()
        wait_for_upstream("127.0.0.1", cls.proxy_port, timeout=10)
        cls.base = f"http://127.0.0.1:{cls.proxy_port}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy.stop()
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def test_audio_response_is_marked(self):
        """The whole point: audio leaving the endpoint carries the marker."""
        from utils.ai_marking import read_ai_marker

        response = urllib.request.urlopen(f"{self.base}/audio", timeout=30)
        body = response.read()

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "out.wav")
            with open(path, "wb") as f:
                f.write(body)
            self.assertIsNotNone(
                read_ai_marker(path),
                "audio served over HTTP carried no EU AI Act Art. 50(2) marking",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_marked_response_has_a_correct_content_length(self):
        """Marking changes the length; a stale header truncates the body."""
        response = urllib.request.urlopen(f"{self.base}/audio", timeout=30)
        body = response.read()
        self.assertEqual(int(response.headers["Content-Length"]), len(body))
        self.assertNotEqual(len(body), len(WAV), "response was not modified at all")

    def test_unmarkable_audio_is_refused_not_forwarded(self):
        """Fail closed on the wire, exactly as on disk."""
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{self.base}/exotic", timeout=30)
        self.assertEqual(caught.exception.code, 502)

    def test_json_passes_through_unchanged(self):
        response = urllib.request.urlopen(f"{self.base}/json", timeout=30)
        self.assertEqual(json.loads(response.read()), {"text": "hello"})

    def test_request_bodies_reach_upstream(self):
        request = urllib.request.Request(f"{self.base}/json", data=b"x" * 4096, method="POST")
        response = urllib.request.urlopen(request, timeout=30)
        self.assertEqual(json.loads(response.read()), {"received": 4096})

    def test_event_streams_are_not_buffered(self):
        """A proxy that buffers SSE turns a working endpoint into a hang.

        Timing is the only way to test this: a buffered stream delivers the
        same bytes, just all at the end. The upstream spaces three events
        0.1s apart, so a streamed read sees the first well before the last.
        """
        import socket

        sock = socket.create_connection(("127.0.0.1", self.proxy_port), timeout=15)
        sock.sendall(b"GET /sse HTTP/1.1\r\nHost: test\r\n\r\n")
        start = time.time()
        arrivals = []
        sock.settimeout(10)
        try:
            while len(arrivals) < 3:
                data = sock.recv(4096)
                if not data:
                    break
                for _ in range(data.decode("utf-8", "replace").count("data:")):
                    arrivals.append(time.time() - start)
        finally:
            sock.close()

        self.assertEqual(len(arrivals), 3, "did not receive all events")
        self.assertGreater(
            arrivals[-1] - arrivals[0],
            0.05,
            f"all events arrived together ({arrivals}) — the stream was buffered",
        )

    def test_protocol_upgrades_are_refused_explicitly(self):
        """Better a named 501 than a tunnel both sides think exists."""
        request = urllib.request.Request(f"{self.base}/json")
        request.add_header("Upgrade", "websocket")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=30)
        self.assertEqual(caught.exception.code, 501)

    def test_upstream_status_codes_survive(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{self.base}/nope", timeout=30)
        self.assertEqual(caught.exception.code, 404)


class TestServerModeRefusesWithoutTheProxy(unittest.TestCase):
    """The CLI must not silently degrade to serving unproxied."""

    def test_refusal_exits_two(self):
        import cli

        with self.assertRaises(SystemExit) as caught:
            cli._refuse_unproxied_server("test reason")
        self.assertEqual(caught.exception.code, 2)

    def test_server_runs_behind_the_proxy_by_default(self):
        import inspect

        import cli

        source = inspect.getsource(cli._run_server)
        self.assertIn("_start_marking_proxy", source)
        # The unproxied path must be reachable only via the attestation.
        self.assertIn("accept_marking_responsibility", source)

    def test_opt_outs_need_the_attestation_here_too(self):
        """--no-watermark reduces provenance on this route as on any other."""
        import inspect

        import cli

        source = inspect.getsource(cli._run_server)
        self.assertIn("_require_marking_attestation", source)

    def test_gui_toggle_uses_the_proxy(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "gui", "main_window.py")
        with open(os.path.abspath(path), encoding="utf-8") as f:
            source = f.read()
        toggle = source.split("def _toggle_server")[1].split("\n    def ")[0]
        self.assertIn("MarkingProxy", toggle)
        self.assertIn("find_free_port", toggle)


class TestServerBindsWhereItWasTold(unittest.TestCase):
    """The binary must listen where start_server says, not where kwargs say.

    Found by running the real binary rather than a stand-in: ``--port`` was
    emitted twice — once by ``start_server`` for the loopback port the proxy
    had chosen, once by ``_append_params`` from the operator's ``--port`` — and
    the last one won. The binary bound the *public* port and served unproxied
    while the proxy waited for an upstream that never arrived. Harmless while
    both sources agreed; a hole in the marking gate the moment they did not.
    """

    def _backend(self, **kwargs):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        return CrispasrBackend(model_id="auto", device="cpu", **kwargs)

    def _server_cmd(self, backend, host, port):
        """Return the command start_server would launch, without launching it.

        The binary is stubbed as well as the launch: this asserts on how the
        command line is *assembled*, which is knowable without a build of
        CrispASR present. Letting it call the real find_crispasr() made these
        tests pass wherever a binary happened to be installed and fail on the
        Windows CI runners, which is a property of the runner rather than of
        the code under test.
        """
        from unittest import mock

        captured = {}

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            raise RuntimeError("stop before launching")

        with mock.patch("utils.crispasr_utils.find_crispasr", return_value="/fake/crispasr"):
            with mock.patch("subprocess.Popen", side_effect=fake_popen):
                try:
                    backend.start_server(host=host, port=port)
                except RuntimeError:
                    pass
        return captured.get("cmd", [])

    def test_port_appears_once_and_is_the_one_requested(self):
        backend = self._backend(port=8080, host="0.0.0.0")
        cmd = self._server_cmd(backend, "127.0.0.1", 49999)

        self.assertEqual(cmd.count("--port"), 1, f"--port emitted twice: {cmd}")
        self.assertEqual(cmd[cmd.index("--port") + 1], "49999")

    def test_host_appears_once_and_is_the_one_requested(self):
        backend = self._backend(port=8080, host="0.0.0.0")
        cmd = self._server_cmd(backend, "127.0.0.1", 49999)

        self.assertEqual(cmd.count("--host"), 1, f"--host emitted twice: {cmd}")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")

    def test_other_kwargs_still_reach_the_binary(self):
        """The exclusion must be surgical, not a blanket suppression."""
        backend = self._backend(port=8080, threads=4)
        cmd = self._server_cmd(backend, "127.0.0.1", 49999)
        self.assertIn("-t", cmd)
        self.assertEqual(cmd[cmd.index("-t") + 1], "4")


class TestAuditLogTailTruncation(unittest.TestCase):
    """The gap a hash chain cannot close on its own."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "biometric.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _record(self, count):
        from utils.audit_log import EVENT_ENROLL, record_event

        for i in range(count):
            record_event(EVENT_ENROLL, speaker=f"person-{i}", path=self.path)

    def test_intact_log_verifies(self):
        from utils.audit_log import verify_chain

        self._record(4)
        report = verify_chain(self.path)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["entries"], 4)
        self.assertEqual(report["anchor"]["entries"], 4)

    def test_truncating_the_tail_is_detected(self):
        """This is the finding: the chain alone still verifies afterwards."""
        from utils.audit_log import verify_chain

        self._record(5)
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(lines[:2])  # drop the last three

        report = verify_chain(self.path)
        self.assertFalse(report["valid"], "tail truncation went undetected")
        self.assertIn("missing from the end", " ".join(report["errors"]))

    def test_replacing_the_final_entry_is_detected(self):
        from utils.audit_log import EVENT_IDENTIFY, _entry_hash, read_events, verify_chain

        self._record(3)
        entries = read_events(self.path)
        entries[-1]["event"] = EVENT_IDENTIFY
        entries[-1]["entry_hash"] = _entry_hash(entries[-1])
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

        report = verify_chain(self.path)
        self.assertFalse(report["valid"], "the anchored head was not checked")

    def test_middle_deletion_still_detected(self):
        """The chain's own guarantee must not have regressed."""
        from utils.audit_log import verify_chain

        self._record(4)
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        del lines[1]
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        self.assertFalse(verify_chain(self.path)["valid"])

    def test_unanchored_log_is_not_called_tampered(self):
        """Logs written before anchoring existed must still verify.

        Treating a missing anchor as evidence of tampering would flag every
        deployment that upgraded, which trains operators to ignore the check.
        """
        from utils.audit_log import verify_chain

        self._record(3)
        os.unlink(f"{self.path}.anchor")
        report = verify_chain(self.path)
        self.assertTrue(report["valid"], report["errors"])
        self.assertIsNone(report["anchor"])

    def test_anchor_holds_no_biometric_payload(self):
        from utils.audit_log import _read_anchor

        self._record(2)
        anchor = _read_anchor(self.path)
        self.assertEqual(set(anchor), {"entries", "head", "updated"})

    def test_anchor_is_a_sibling_not_part_of_the_log(self):
        """An anchor inside the log would be truncated by the same edit."""
        from utils.audit_log import anchor_path

        self.assertNotEqual(anchor_path(self.path), self.path)
        self._record(1)
        self.assertTrue(os.path.isfile(anchor_path(self.path)))


class TestSyntheticTextDisclosure(unittest.TestCase):
    def test_cli_discloses_on_stderr_not_stdout(self):
        """stdout stays pipeable; the notice must not corrupt the payload."""
        import contextlib

        import cli

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli._disclose_synthetic_text()

        self.assertEqual(out.getvalue(), "", "disclosure leaked into stdout")
        self.assertIn("machine-translation", err.getvalue())
        self.assertIn("Art. 50(4)", err.getvalue())

    def test_gui_notice_exists_in_both_languages(self):
        from utils.i18n import t

        for locale in ("en", "de"):
            notice = t("notice.synthetic_text", locale=locale)
            self.assertTrue(notice, f"no synthetic-text notice for {locale}")
            self.assertIn("50(4)", notice)

    def test_gui_shows_the_notice_beside_the_result(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "gui", "main_window.py")
        with open(os.path.abspath(path), encoding="utf-8") as f:
            source = f.read()
        handler = source.split("def _on_translation_finished")[1].split("\n    def ")[0]
        self.assertIn("notice.synthetic_text", handler)


if __name__ == "__main__":
    unittest.main()
