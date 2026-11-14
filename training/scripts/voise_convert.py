"""Speak it yourself, hear it in a consenting person's voice: a recording in -> the same words, rhythm, laughs and
surprise, in the person's voice out.

Text-to-speech has to invent the delivery, which is why its laughs and emotions sound made up. Here the delivery is
the speaker's own: kNN-VC (Baas et al., 2023; experiments/knnvc_2026-10-24) keeps every WavLM frame's timing and
replaces its content with the nearest frames of the person's real speech (their clips in the learning pool), then
a vocoder turns that back into audio. Nothing is trained; the more of the person's speech the pool holds (about
10 minutes is enough), the closer the voice.

Same rules as the TTS tool: the target person's consent record must be complete and unrevoked, the output is a
file (never a live stream or a call) named SYNTHETIC_*, it carries the inaudible watermark, and a sidecar records the
consent and the source. The source should be your own recording.

  python voise_convert.py --consent runs/consent/mafia1.json --person mafia_abood --src my_note.ogg --out runs/convert
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_clone_tts as V  # noqa: E402

KNNVC_DIR = os.path.join(V.ROOT, "models", "knnvc")
SR = 16000


def load_audio(path):
    """Any format ffmpeg reads (WhatsApp .ogg/.opus, .mp3, .m4a, .wav) -> 16 kHz mono float32."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


class Converter:
    def __init__(self, weights=KNNVC_DIR, k=4):
        os.environ["KNNVC_W"] = weights
        sys.path.insert(0, os.path.join(V.ROOT, "experiments", "knnvc_2026-10-24"))
        import knnvc_lib as K
        self.K, self.k = K, k
        self.wavlm, self.gen = K.load_wavlm(), K.load_gen()
        self.sets = {}

    def matching_set(self, person_dir):
        """WavLM features of the person's real clips, cached next to them (recomputed when clips change)."""
        import torch
        wavs = sorted(glob.glob(os.path.join(person_dir, "*.wav")))
        if not wavs:
            raise SystemExit(f"no recordings for this person in {person_dir}")
        key = (person_dir, len(wavs), max(os.path.getmtime(w) for w in wavs))
        if key in self.sets:
            return self.sets[key]
        cache = os.path.join(person_dir, ".knnvc_matching.pt")
        if os.path.exists(cache):
            c = torch.load(cache, map_location="cpu", weights_only=True)
            if c.get("n") == len(wavs) and c.get("mtime") == key[2]:
                self.sets[key] = (c["feats"], c["seconds"])
                return self.sets[key]
        feats = torch.cat([self.K.wavlm_feats(self.wavlm, sf.read(w, dtype="float32")[0]) for w in wavs])
        seconds = round(sum(sf.info(w).duration for w in wavs), 1)
        torch.save({"n": len(wavs), "mtime": key[2], "feats": feats, "seconds": seconds}, cache)
        self.sets[key] = (feats, seconds)
        return self.sets[key]

    def convert(self, y16, person_dir):
        import torch
        mset, seconds = self.matching_set(person_dir)
        with torch.no_grad():
            q = self.K.wavlm_feats(self.wavlm, y16)
            out = self.gen(self.K.knn(q, mset, k=self.k)[None])[0].numpy()
        return out, seconds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--consent", required=True, help="the TARGET person's consent record")
    ap.add_argument("--person", required=True, help="the person's folder name in the learning pool")
    ap.add_argument("--src", required=True, help="your recording (any format)")
    ap.add_argument("--out", default=os.path.join(V.RUNS, "convert"))
    ap.add_argument("--pool", default=os.path.join(V.RUNS, "learn", "pool"))
    ap.add_argument("--asr-dir", default="models/whisper-turbo-dialectal-v2-ct2")
    ap.add_argument("--sv-dir", default="models/ecapa-voxceleb")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-watermark", action="store_true")
    a = ap.parse_args(argv)

    consent, consent_sha = V.load_consent(a.consent)          # refuses a missing, incomplete or revoked record
    person = V.safe_slug(a.person)
    person_dir = os.path.join(a.pool, person)
    out = V.check_out_dir(a.out)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wav_path, meta_path = V.output_paths(out, person, 0, stamp)

    src = load_audio(a.src)
    conv = Converter(k=a.k)
    y, seconds = conv.convert(src, person_dir)
    y = y / max(1e-6, float(np.abs(y).max())) * 0.9
    wm = None
    if not a.no_watermark:
        y, wm = V.watermark(y, SR)
    sf.write(wav_path, y, SR, subtype="PCM_16")

    said, heard = V.transcribe(src, a.asr_dir), V.transcribe(y, a.asr_dir)
    sim = None
    enc = V.load_speaker_encoder(a.sv_dir) if a.sv_dir and os.path.isdir(a.sv_dir) else None
    if enc is not None:
        clips = sorted(glob.glob(os.path.join(person_dir, "*.wav")))[:20]
        c = sum(V.speaker_embedding(enc, sf.read(w, dtype="float32")[0]) for w in clips)
        c = c / c.norm()
        sim = round(float(V.speaker_embedding(enc, y) @ c), 3)
    json.dump(dict(synthetic=True, method="kNN-VC voice conversion (WavLM-Large layer 6, prematched HiFi-GAN), k=%d" % a.k,
                   voice_owner=person, consent_record=a.consent, consent_sha256=consent_sha, consent_scope=consent["scope"],
                   source=os.path.basename(a.src), matching_seconds=seconds, said=said, heard=heard,
                   letter_error_vs_source=round(V.cer(said, heard), 3), similarity_to_person=sim,
                   watermark=wm, created_utc=stamp), open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{wav_path} (voice from {seconds} s of {person}'s speech; similarity {sim}; heard: {heard})")


if __name__ == "__main__":
    main()
