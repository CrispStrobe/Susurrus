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

    def test_streaming_synthesis_is_refused_not_silently_buffered(self):
        """/v1/audio/speech accepts stream=true and pushes audio per sentence.

        Marking needs the finished samples, so the proxy would have to buffer
        the whole stream — turning a streaming endpoint into a non-streaming
        one for reasons invisible from the client. Refuse and say so.
        """
        request = urllib.request.Request(
            f"{self.base}/v1/audio/speech",
            data=b'{"input":"hello","stream":true,"response_format":"pcm"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=30)
        self.assertEqual(caught.exception.code, 502)
        self.assertIn("streaming", caught.exception.read().decode())

    def test_non_streaming_synthesis_still_works(self):
        """The refusal must key on the flag, not on the endpoint."""
        request = urllib.request.Request(
            f"{self.base}/json",
            data=b'{"input":"hello","stream":false}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=30)
        self.assertEqual(response.status, 200)

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


class TestCompleteMarkingDoesNotRestack(unittest.TestCase):
    """Audio the binary already marked must not be marked over the top.

    Found by pointing the proxy at a real TTS server: the binary applies the
    in-sample watermark itself (measured 0.815 on Piper output, above the 0.78
    threshold), and the proxy was running the full apply_provenance pipeline
    over it — a second comb on the first, ~37-41 dB SNR for a mark that already
    verified, plus a second C2PA manifest. The local CrispASR routes have
    avoided this since they were written; the proxy had to learn it.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "audio.wav")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, data, rate=22050):
        import soundfile as sf

        sf.write(self.path, data, rate, subtype="PCM_16")

    def _samples(self):
        import soundfile as sf

        data, _ = sf.read(self.path, dtype="float32")
        return data

    def _speechish(self, seconds=3.0, rate=22050):
        import numpy as np

        rng = __import__("numpy").random.default_rng(3)
        n = int(seconds * rate)
        x = np.convolve(rng.standard_normal(n), np.ones(24) / 24, mode="same")
        return (x / max(abs(x).max(), 1e-9) * 0.3).astype("float32")

    def test_existing_watermark_is_not_re_embedded(self):
        """The samples must come out byte-identical."""
        import numpy as np

        from utils import spread_spectrum as ss
        from utils.provenance import complete_marking

        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")

        marked = ss.embed(self._speechish())
        self._write(marked)
        before = self._samples()

        complete_marking(self.path, model="test")

        after = self._samples()
        self.assertEqual(len(before), len(after))
        self.assertTrue(
            np.array_equal(before, after),
            "samples were modified — a second watermark was stacked on the first",
        )

    def test_the_declarative_marker_is_still_applied(self):
        """Not re-watermarking must not mean not marking."""
        from utils import spread_spectrum as ss
        from utils.ai_marking import read_ai_marker
        from utils.provenance import complete_marking

        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")

        self._write(ss.embed(self._speechish()))
        complete_marking(self.path, model="test")
        self.assertIsNotNone(read_ai_marker(self.path))

    def test_missing_watermark_is_added(self):
        """Gaps are still filled; only what is present is left alone."""
        from utils import spread_spectrum as ss
        from utils.provenance import complete_marking

        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")

        plain = self._speechish()
        self._write(plain)
        self.assertLess(ss.detect(plain), ss.DETECTION_THRESHOLD)

        complete_marking(self.path, model="test")

        self.assertGreaterEqual(ss.detect(self._samples()), ss.DETECTION_THRESHOLD)

    def test_it_still_fails_closed(self):
        """An unmarkable container is refused and deleted, as everywhere else."""
        from utils.provenance import ProvenanceError, complete_marking

        exotic = os.path.join(self.tmpdir, "audio.au")
        with open(exotic, "wb") as f:
            f.write(b"\x2esnd" + b"\x00" * 64)

        with self.assertRaises(ProvenanceError):
            complete_marking(exotic, options={"no_c2pa": True, "no_watermark": True})
        self.assertFalse(os.path.exists(exotic))

    def test_marker_survives_the_watermark_step(self):
        """Layer order: the watermarker rewrites the file, so it must go first.

        Marking before watermarking loses the marker — soundfile rebuilds the
        file from its samples and the appended RIFF chunk goes with it. The
        result dict still said ``marker: True`` while the disk said otherwise,
        which is the failure mode the whole verify-the-file discipline exists
        to catch. Only reproducible on audio that needs a watermark added, so
        it hid behind every fixture that already had one.
        """
        from utils.ai_marking import read_ai_marker
        from utils.provenance import complete_marking

        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")

        self._write(self._speechish())  # no watermark: forces the embed path
        result = complete_marking(self.path, model="test")

        self.assertTrue(result["watermark"], "watermark was not applied")
        self.assertTrue(result["marker"], "result claims no marker")
        self.assertIsNotNone(
            read_ai_marker(self.path),
            "result claimed a marker the file does not carry — the watermark "
            "step rewrote the file and dropped it",
        )

    def test_the_proxy_uses_it(self):
        import inspect

        from utils import marking_proxy

        source = inspect.getsource(marking_proxy.MarkingProxy.mark_audio)
        self.assertIn("complete_marking(path", source)
        self.assertNotIn("apply_provenance(", source)


class TestRawSampleResponsesAreMarked(unittest.TestCase):
    """Container-less synthesis output must not escape on a Content-Type.

    ``response_format: "f32"`` returns raw float32 as
    ``application/octet-stream``. That is not ``audio/*``, so the proxy
    classified it as "not audio" and forwarded it untouched — synthetic audio
    served unmarked by the very route the proxy guards. Observed against the
    real server before it was fixed. The endpoint, not the label, decides
    whether a payload is synthetic.
    """

    def test_synthesis_paths_are_recognised(self):
        from utils.marking_proxy import is_synthesis_path

        for path in ("/v1/audio/speech", "/v1/audio/speech-to-speech", "/v1/audio/speech?x=1"):
            self.assertTrue(is_synthesis_path(path), path)
        for path in ("/v1/audio/transcriptions", "/v1/models", "/health", ""):
            self.assertFalse(is_synthesis_path(path), path)

    def _proxy(self, **options):
        from utils.marking_proxy import MarkingProxy

        return MarkingProxy("127.0.0.1", 0, "127.0.0.1", 0, options=options)

    def _speechish(self, n=48000):
        import numpy as np

        rng = np.random.default_rng(11)
        x = np.convolve(rng.standard_normal(n), np.ones(24) / 24, mode="same")
        return (x / max(abs(x).max(), 1e-9) * 0.3).astype("float32")

    def test_unmarked_float32_gets_a_watermark(self):
        import numpy as np

        from utils import spread_spectrum as ss

        plain = self._speechish()
        self.assertLess(ss.detect(plain), ss.DETECTION_THRESHOLD)

        out = self._proxy().mark_raw_samples(plain.tobytes(), "<f4")
        self.assertIsNotNone(out, "raw float32 was refused instead of marked")

        marked = np.frombuffer(out, dtype="<f4")
        self.assertEqual(len(marked), len(plain))
        self.assertGreaterEqual(ss.detect(np.ascontiguousarray(marked)), ss.DETECTION_THRESHOLD)

    def test_already_marked_float32_is_passed_through_untouched(self):
        """Same no-restacking rule as the container paths."""
        from utils import spread_spectrum as ss

        pre = ss.embed(self._speechish())
        body = pre.tobytes()
        out = self._proxy().mark_raw_samples(body, "<f4")
        self.assertEqual(out, body, "a second comb was stacked on the first")

    def test_int16_pcm_round_trips_at_the_same_width(self):
        import numpy as np

        from utils import spread_spectrum as ss

        pcm = (self._speechish() * 32767).astype("<i2")
        out = self._proxy().mark_raw_samples(pcm.tobytes(), "<i2")
        self.assertIsNotNone(out)
        self.assertEqual(len(out), len(pcm.tobytes()), "sample width changed")

        back = np.frombuffer(out, dtype="<i2").astype("float32") / 32768
        self.assertGreaterEqual(ss.detect(np.ascontiguousarray(back)), ss.DETECTION_THRESHOLD)

    def test_opting_out_of_the_watermark_is_the_only_way_past(self):
        """Nothing else can mark a container-less payload."""
        plain = self._speechish()
        body = plain.tobytes()
        self.assertEqual(self._proxy(no_watermark=True).mark_raw_samples(body, "<f4"), body)

    def test_request_format_is_parsed_from_the_body(self):
        from utils.marking_proxy import _requested_format

        self.assertEqual(_requested_format(b'{"input":"x","response_format":"F32"}'), "f32")
        self.assertIsNone(_requested_format(b'{"input":"x"}'))
        self.assertIsNone(_requested_format(b"not json"))
        self.assertIsNone(_requested_format(b"[1,2,3]"))

    def test_streaming_flag_is_parsed(self):
        from utils.marking_proxy import _parse_synthesis_request

        self.assertEqual(
            _parse_synthesis_request(b'{"response_format":"pcm","stream":true}'), ("pcm", True)
        )
        self.assertEqual(_parse_synthesis_request(b'{"response_format":"wav"}'), ("wav", False))
        self.assertEqual(_parse_synthesis_request(b"garbage"), (None, False))


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
