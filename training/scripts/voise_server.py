"""Keep Voise's models loaded so each sentence costs only its own generation.

Loading Lahgtna, the learned version, Whisper and preparing a voice took about 5 minutes on every run; this
process pays that once and then serves sentences. It listens on 127.0.0.1 only (this computer, never the network).

  GET  /             a simple page to pick a person, type a sentence and listen (for demos)
  POST /speak        {"person": "khaled", "text": "...", "fast": false, "candidates": 3}  ->  {"wav": ..., "heard": ...}
  GET  /people       people with a complete, unrevoked consent record and recordings
  GET  /history/<p>  that person's latest outputs
  GET  /audio/<p>/<SYNTHETIC_*.wav>
  GET  /health

Every request re-reads the person's consent record (a revoked record is refused at once), and the learned version
is reloaded automatically when voise_learn.py writes a new generation.

Usage: python voise_server.py [--port 8765] [--model-dir ...] [--asr-dir ...] [--adapter runs/learn/adapter.pt]
"""
import argparse
import glob
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
AUDIO = re.compile(r"^SYNTHETIC_[A-Za-z0-9_]+\.wav$")


def people():
    """People the page may offer: a complete, unrevoked consent record and at least one recording."""
    out = []
    for c in sorted(glob.glob(os.path.join(V.RUNS, "consent", "*.json"))):
        person = os.path.basename(c)[:-5]
        if not PERSON.match(person) or not glob.glob(os.path.join(V.RUNS, "voices", person, "*.wav")):
            continue
        try:
            rec, _ = V.load_consent(c)
        except SystemExit:
            continue
        out.append(dict(person=person, owner=rec["voice_owner"], scope=rec["scope"], training=bool(rec.get("training"))))
    return out


def history(person, n=8):
    items = []
    for j in sorted(glob.glob(os.path.join(V.RUNS, "voices", person, "output", "SYNTHETIC_*.json")),
                    key=os.path.getmtime, reverse=True)[:n]:
        if os.path.exists(j[:-5] + ".wav"):
            items.append(dict(wav=j[:-5] + ".wav", text=json.load(open(j, encoding="utf-8")).get("text", "")))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-dir", default="models/lahgtna-omnivoice-v2")
    ap.add_argument("--asr-dir", default="models/whisper-turbo-dialectal-v2-ct2")
    ap.add_argument("--adapter", default=os.path.join(V.RUNS, "learn", "adapter.pt"))
    ap.add_argument("--sv-dir", help="speaker encoder (speechbrain/spkrec-ecapa-voxceleb) for choosing references")
    ap.add_argument("--mos", help="UTMOS22-strong checkpoint, to report each take's predicted naturalness")
    a = ap.parse_args(argv)

    t0 = time.time()
    eng = V.Engine(a.model_dir, a.asr_dir, a.adapter, a.sv_dir, mos_path=a.mos)
    V.transcribe(__import__("numpy").zeros(V.SR, "float32"), a.asr_dir)       # load Whisper now, not on request 1
    state = {"adapter_mtime": os.path.getmtime(a.adapter) if os.path.exists(a.adapter) else None}
    lock = threading.Lock()
    print(f"ready in {time.time() - t0:.0f}s: open http://127.0.0.1:{a.port} in your browser "
          f"(learned version: {eng.adapter})", flush=True)

    def refresh_adapter():
        m = os.path.getmtime(a.adapter) if os.path.exists(a.adapter) else None
        if m != state["adapter_mtime"]:
            global_eng = V.Engine(a.model_dir, a.asr_dir, a.adapter, a.sv_dir, mos_path=a.mos)
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

        def send_file(self, path, ctype):
            data = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def trusted(self, post=False):
            """Only this computer's own page may use the engine. The Host check stops DNS-rebinding sites; for
            POST, a JSON content type (which a cross-site form cannot send without a preflight this server never
            answers) and, when present, a matching Origin stop another website open in the browser from making
            voices through it."""
            ok_hosts = {f"127.0.0.1:{a.port}", f"localhost:{a.port}"}
            if self.headers.get("Host", "") not in ok_hosts:
                return False
            if post:
                origin = self.headers.get("Origin")
                if origin is not None and origin not in {f"http://{h}" for h in ok_hosts}:
                    return False
                if not self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json":
                    return False
            return True

        def do_GET(self):
            if not self.trusted():
                return self.reply(403, {"error": "forbidden"})
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                return self.send_file(os.path.join(HERE, "voise_ui.html"), "text/html; charset=utf-8")
            if parts == ["health"]:
                return self.reply(200, {"ok": True, "adapter": eng.adapter, "voices_cached": len(eng.voices)})
            if parts == ["people"]:
                return self.reply(200, {"people": people()})
            if len(parts) == 2 and parts[0] == "history" and PERSON.match(parts[1]):
                return self.reply(200, {"items": history(parts[1])})
            if len(parts) == 3 and parts[0] == "audio" and PERSON.match(parts[1]) and AUDIO.match(parts[2]):
                path = os.path.join(V.RUNS, "voices", parts[1], "output", parts[2])
                if os.path.isfile(path):
                    return self.send_file(path, "audio/wav")
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.trusted(post=True):
                return self.reply(403, {"error": "forbidden"})
            if self.path != "/speak":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                person, text = str(req.get("person", "")), str(req.get("text", "")).strip()
                if not PERSON.match(person) or not text or len(text) > 1500:
                    return self.reply(400, {"error": "need person (a-z, 0-9, _) and text (1-1500 characters)"})
                with lock:
                    refresh_adapter()
                    t = time.time()
                    wav, best, cands = eng.say(os.path.join(V.RUNS, "consent", f"{person}.json"),
                                               os.path.join(V.RUNS, "voices", person), text,
                                               os.path.join(V.RUNS, "voices", person, "output"),
                                               candidates=int(req.get("candidates", 3)),
                                               steps=16 if req.get("fast") else 32, who=person)
                self.reply(200, {"wav": wav, "heard": best["heard"], "takes": len(cands), "seconds": round(time.time() - t),
                                 "naturalness": best.get("mos")})
            except SystemExit as e:                     # consent refused, missing recordings, ...
                self.reply(403, {"error": str(e)})
            except Exception as e:                       # keep serving after a bad request
                self.reply(500, {"error": repr(e)})

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
