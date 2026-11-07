"""voise_server: only this computer's own page may use the engine (no model is loaded)."""
import http.client
import json
import os
import socket
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import voice_clone_tts as V  # noqa: E402
import voise_server as S  # noqa: E402


class FakeEngine:
    def __init__(self, *a, **k):
        self.adapter, self.voices = None, {}

    def say(self, consent, ref_dir, text, out, **k):
        return "/x/SYNTHETIC_test.wav", {"heard": text}, [1]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        V.Engine, V.transcribe = FakeEngine, lambda *a, **k: ""
        cls.port = free_port()
        threading.Thread(target=S.main, args=(["--port", str(cls.port), "--adapter", "/nonexistent"],), daemon=True).start()
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=0.2).close(); break
            except OSError:
                time.sleep(0.1)

    def post(self, body, ctype="application/json", host=None, origin=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Content-Type": ctype, "Host": host or f"127.0.0.1:{self.port}"}
        if origin:
            h["Origin"] = origin
        c.request("POST", "/speak", body=body, headers=h)
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}")

    def test_own_page_json_request_is_served(self):
        st, r = self.post(json.dumps({"person": "test", "text": "مرحبا"}), origin=f"http://127.0.0.1:{self.port}")
        self.assertEqual(st, 200)
        self.assertEqual(r["heard"], "مرحبا")

    def test_cross_site_form_post_is_refused(self):
        st, _ = self.post(json.dumps({"person": "test", "text": "x"}), ctype="text/plain")
        self.assertEqual(st, 403)

    def test_other_origin_is_refused(self):
        st, _ = self.post(json.dumps({"person": "test", "text": "x"}), origin="http://evil.example")
        self.assertEqual(st, 403)

    def test_dns_rebinding_host_is_refused(self):
        st, _ = self.post(json.dumps({"person": "test", "text": "x"}), host="evil.example:80")
        self.assertEqual(st, 403)

    def test_bad_person_name_is_rejected(self):
        st, _ = self.post(json.dumps({"person": "../etc", "text": "x"}))
        self.assertEqual(st, 400)


if __name__ == "__main__":
    unittest.main()
