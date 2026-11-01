"""Keep Voise's models loaded so each sentence costs only its own generation.

Loading Lahgtna, the learned version, Whisper and preparing a voice took about 5 minutes on every run; this
process pays that once and then serves sentences. It listens on 127.0.0.1 only (this computer, never the network).

  POST /speak   {"person": "khaled", "text": "...", "fast": false, "candidates": 3}  ->  {"wav": ..., "heard": ...}
  GET  /health

Every request re-reads the person's consent record (a revoked record is refused at once), and the learned version
is reloaded automatically when voise_learn.py writes a new generation.

Usage: python voise_server.py [--port 8765] [--model-dir ...] [--asr-dir ...] [--adapter runs/learn/adapter.pt]
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_clone_tts as V  # noqa: E402

PERSON = re.compile(r"^[a-z0-9_]{1,64}$")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-dir", default="models/lahgtna-omnivoice-v2")
    ap.add_argument("--asr-dir", default="models/whisper-turbo-dialectal-v2-ct2")
    ap.add_argument("--adapter", default=os.path.join(V.RUNS, "learn", "adapter.pt"))
    a = ap.parse_args(argv)

    t0 = time.time()
    eng = V.Engine(a.model_dir, a.asr_dir, a.adapter)
    V.transcribe(__import__("numpy").zeros(V.SR, "float32"), a.asr_dir)       # load Whisper now, not on request 1
    state = {"adapter_mtime": os.path.getmtime(a.adapter) if os.path.exists(a.adapter) else None}
    lock = threading.Lock()
    print(f"ready in {time.time() - t0:.0f}s on http://127.0.0.1:{a.port} (learned version: {eng.adapter})", flush=True)

    def refresh_adapter():
        m = os.path.getmtime(a.adapter) if os.path.exists(a.adapter) else None
        if m != state["adapter_mtime"]:
            global_eng = V.Engine(a.model_dir, a.asr_dir, a.adapter)
            eng.model, eng.adapter, eng.voices = global_eng.model, global_eng.adapter, {}
            state["adapter_mtime"] = m
            print("learned version changed: reloaded", eng.adapter, flush=True)

    class H(BaseHTTPRequestHandler):
        def reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"ok": True, "adapter": eng.adapter, "voices_cached": len(eng.voices)})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/speak":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                person, text = str(req.get("person", "")), str(req.get("text", "")).strip()
                if not PERSON.match(person) or not text or len(text) > 600:
                    return self.reply(400, {"error": "need person (a-z, 0-9, _) and text (1-600 characters)"})
                with lock:
                    refresh_adapter()
                    t = time.time()
                    wav, best, cands = eng.say(os.path.join(V.RUNS, "consent", f"{person}.json"),
                                               os.path.join(V.RUNS, "voices", person), text,
                                               os.path.join(V.RUNS, "voices", person, "output"),
                                               candidates=int(req.get("candidates", 3)),
                                               steps=16 if req.get("fast") else 32)
                self.reply(200, {"wav": wav, "heard": best["heard"], "takes": len(cands), "seconds": round(time.time() - t)})
            except SystemExit as e:                     # consent refused, missing recordings, ...
                self.reply(403, {"error": str(e)})
            except Exception as e:                       # keep serving after a bad request
                self.reply(500, {"error": repr(e)})

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
