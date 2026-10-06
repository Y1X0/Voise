#!/usr/bin/env python3
"""TRAINING_READINESS implementation tests (no GPU, no real datasets; synthetic audio only):
interrupted-training resume (bit-exact), checkpoint provenance metadata, scheduler/scaler/RNG/
monitor state, GPU profiles, the strict dataset gate, the streaming sampler + feature cache,
the real-run driver end-to-end on CPU, MLS integration on a synthetic official layout, the
Arabic consented-recording importer, the English+Arabic mix, the locked acceptance evaluator
and the Android benchmark report.

  python3 training/tests/test_training_ready.py
"""
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import soundfile as sf
import torch

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SR = 16000
HAVE_SMOKE = os.path.isdir(os.path.join(ROOT, "models", "librosa-data"))


def voiced(sec, f0=140.0, seed=0):
    t = np.arange(int(sec * SR)) / SR
    rng = np.random.default_rng(seed)
    am = np.interp(t, np.linspace(0, sec, 12), rng.uniform(0.1, 1.0, 12))       # utterance-specific envelope
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * am
    return (0.2 * x / np.abs(x).max() + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def smoke_trainer(out):
    from trainers.config import load_config
    from trainers.smoke import build
    cfg = load_config(os.path.join(TRAINING, "configs", "smoke.yaml"))
    cfg.data["segment_seconds"], cfg.data["batch_size"] = 1.0, 2
    tr, _ = build(cfg, out, 0)
    return tr


@unittest.skipUnless(HAVE_SMOKE, "smoke data not fetched")
class ResumeAndMetadata(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_interrupted_training_resumes_bit_exact(self):
        from trainers.checkpoint import latest_valid
        a = smoke_trainer(os.path.join(self.d, "a"))
        a.run_stage("content_distillation", 6, val_every=100, ckpt_every=2)
        b = smoke_trainer(os.path.join(self.d, "b"))
        real, calls = b.step_content, {"n": 0}

        def dies(batch):
            calls["n"] += 1
            if calls["n"] == 5:
                raise KeyboardInterrupt("simulated disconnect")
            return real(batch)
        b.step_content = dies
        with self.assertRaises(KeyboardInterrupt):
            b.run_stage("content_distillation", 6, val_every=100, ckpt_every=2)
        c = smoke_trainer(os.path.join(self.d, "b"))
        ck = latest_valid(os.path.join(self.d, "b", "content_distillation"))
        self.assertTrue(ck.endswith("step_4.pt") or ck.endswith("last.pt"))
        c.load(ck)
        self.assertEqual(c.step, 4)
        c.run_stage("content_distillation", 6, val_every=100, ckpt_every=2, resume=True)
        diff = max(float((x - y).abs().max()) for x, y in zip(a.model.state_dict().values(), c.model.state_dict().values()))
        self.assertEqual(diff, 0.0)
        self.assertEqual(a.sched_g.last_epoch, c.sched_g.last_epoch)

    def test_checkpoint_contains_state_and_provenance(self):
        tr = smoke_trainer(self.d)
        tr.run_info = {"config_sha256": "c" * 64, "manifest_sha256": {"m.jsonl": "d" * 64}, "git_sha": "e" * 40,
                       "git_dirty": False, "env_lock_sha256": "f" * 64, "env": {"gpu": None}}
        tr.run_stage("content_distillation", 2, val_every=100, ckpt_every=2)
        c = torch.load(tr.ckpt_path("last"), weights_only=False)
        for k in ("opt_g", "opt_d", "sched_g", "sched_d", "scaler_g", "scaler_d", "torch_rng", "monitor", "meta"):
            self.assertIn(k, c)
        side = json.load(open(tr.ckpt_path("last") + ".json"))
        for k in ("config_sha256", "manifest_sha256", "git_sha", "model_version", "model_cfg_sha256", "precision", "sha256"):
            self.assertIn(k, side)
        self.assertEqual(side["git_sha"], "e" * 40)
        self.assertEqual(side["manifest_sha256"], {"m.jsonl": "d" * 64})

    def test_model_version_mismatch_refused(self):
        tr = smoke_trainer(self.d)
        tr.run_stage("content_distillation", 2, val_every=100, ckpt_every=2)
        c = torch.load(tr.ckpt_path("last"), weights_only=False)
        c["meta"]["model_version"] = "StreamAnon/0"
        p = os.path.join(self.d, "old.pt")
        torch.save(c, p)
        with self.assertRaises(ValueError):
            smoke_trainer(self.d).load(p)


class GpuProfiles(unittest.TestCase):
    def test_profiles_load_label_and_apply(self):
        from trainers import gpu_profile as GP
        from trainers.config import load_config
        for name, prec in (("t4_16gb", "fp16"), ("a100_40gb", "bf16"), ("a100_80gb", "bf16")):
            p = GP.load(name)
            self.assertEqual(p["precision"]["value"], prec)
            for path, v in GP._walk(p):
                self.assertIn(v["label"], GP.LABELS, path)
                if "hours" in path or "seconds_per_step" in path:
                    self.assertNotEqual(v["label"], "MEASURED", path)     # estimates are never "verified compute"
            cfg = load_config(os.path.join(TRAINING, "configs", "stream_anon_s.yaml"))
            kw = GP.apply(cfg, p)
            self.assertEqual(kw["batch_size"], 16)
            self.assertEqual(cfg.model.n_speakers, 8192)               # model untouched except data/optim knobs

    def test_bad_profiles_rejected(self):
        import yaml
        from trainers import gpu_profile as GP
        p = yaml.safe_load(open(os.path.join(GP.PROFILE_DIR, "t4_16gb.yaml")))
        d = tempfile.mkdtemp()
        try:
            for mut in ({"grad_accumulation": {"value": 2, "label": "ESTIMATED"}},
                        {"batch_size": {"value": 16, "label": "GUESSED"}}, {"batch_size": 16}):
                q = dict(p, **mut)
                f = os.path.join(d, "x.yaml")
                yaml.safe_dump(q, open(f, "w"))
                with self.assertRaises(GP.ProfileError):
                    GP.load(f)
        finally:
            shutil.rmtree(d)


class DatasetGate(unittest.TestCase):
    def test_statuses(self):
        from datasets import gate as G
        self.assertEqual(G.gate_status("mls_en"), "COMMERCIAL_PENDING")          # not downloaded / verified
        self.assertEqual(G.gate_status("vctk"), "COMMERCIAL_PENDING")
        self.assertEqual(G.gate_status("qasr"), "RESEARCH_ONLY")
        self.assertEqual(G.gate_status("masc"), "NOT_ELIGIBLE")
        self.assertEqual(G.gate_status("yodas_ar_manual"), "NOT_ELIGIBLE")
        self.assertEqual(G.gate_status("own_recordings"), "CONSENT_REQUIRED")
        self.assertEqual(G.gate_status("fleurs"), "NOT_ELIGIBLE")               # evaluation only by rule
        self.assertEqual(G.gate_status("totally_unknown"), "NOT_ELIGIBLE")
        self.assertEqual(G.gate_status("librosa_example"), "COMMERCIAL_VERIFIED")  # pinned smoke excerpts
        self.assertEqual(G.gate_status("mls_en", prov={"mls_en": {"archive_sha256": "a", "licence_text_sha256": "b"}}),
                         "COMMERCIAL_VERIFIED")
        with self.assertRaises(G.GateError):
            G.assert_trainable({"mls_en"}, "commercial")
        G.assert_trainable({"mls_en", "qasr"}, "research")
        with self.assertRaises(G.GateError):
            G.assert_trainable({"masc"}, "research")

    @unittest.skipUnless(HAVE_SMOKE, "smoke data not fetched")
    def test_smoke_provenance_hashes_match_files(self):
        import hashlib
        p = json.load(open(os.path.join(ROOT, "data", "provenance.json")))["verified"]["librosa_example"]
        h = hashlib.sha256()
        for f in ("198-209-0000", "3436-172162-0000", "5703-47212-0000"):
            h.update(open(os.path.join(ROOT, "models", "librosa-data", "audio", f + ".hq.ogg"), "rb").read())
        self.assertEqual(h.hexdigest(), p["archive_sha256"])


def write_feature_fixture(d, speakers=3, sec=6.0, corpus="librosa_example", n_units=32):
    from datasets import feature_cache as FC
    rows = []
    os.makedirs(os.path.join(d, "wav"), exist_ok=True)
    os.makedirs(os.path.join(d, "units"), exist_ok=True)
    for s in range(speakers):
        for u in range(2):
            p = os.path.join(d, "wav", f"s{s}_{u}.wav")
            sf.write(p, voiced(sec, 110 + 30 * s, s * 10 + u), SR)
            split = "valid" if s == speakers - 1 else "train"
            rows.append({"path": p, "speaker": f"{corpus}:{s}", "session": str(u), "corpus": corpus, "language": "en",
                         "duration": sec, "sr": SR, "split": split})
    for r in rows:
        np.save(os.path.join(d, "units", FC.row_key(r) + ".npy"),
                np.random.default_rng(1).integers(0, n_units, int(r["duration"] * 50)))
    idx, digest = FC.build(rows, os.path.join(d, "cache"), os.path.join(d, "units"))
    return idx, digest, rows


class StreamingData(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_feature_cache_idempotent_and_sampler_deterministic(self):
        from datasets.stream_sampler import StreamingSegmentSampler, prefetching_batches
        idx, digest, rows = write_feature_fixture(self.d)
        from datasets import feature_cache as FC
        _, digest2 = FC.build(rows, os.path.join(self.d, "cache"), os.path.join(self.d, "units"))
        self.assertEqual(digest, digest2)
        s = StreamingSegmentSampler(idx, "train", 1.0, seed=3, licence_path="smoke")
        b1, b2 = s.batch(7, 4), s.batch(7, 4)
        for k in b1:
            self.assertTrue(torch.equal(b1[k], b2[k]), k)
        self.assertEqual(tuple(b1["wav"].shape), (4, 16000))
        self.assertEqual(tuple(b1["prosody"].shape), (4, 100, 3))
        self.assertEqual(tuple(b1["units"].shape), (4, 100))
        par = list(prefetching_batches(s, 5, 3, 2, num_workers=2))
        ser = [s.batch(5 + i, 2) for i in range(3)]
        for p, q in zip(par, ser):
            self.assertTrue(torch.equal(p["wav"], q["wav"]))

    def test_sampler_refuses_ineligible_corpus(self):
        from datasets.gate import GateError
        from datasets.stream_sampler import StreamingSegmentSampler
        idx, _, _ = write_feature_fixture(self.d, corpus="masc")
        with self.assertRaises(GateError):
            StreamingSegmentSampler(idx, "train", 1.0, licence_path="commercial")
        with self.assertRaises(GateError):
            StreamingSegmentSampler(idx, "train", 1.0, licence_path="research")

    def test_mix_weights_follow_language_shares(self):
        from datasets.stream_sampler import StreamingSegmentSampler
        idx, _, _ = write_feature_fixture(self.d, speakers=4)
        ents = [json.loads(l) for l in open(idx)]
        for e in ents:
            e["mix_weight"] = 0.9 if e["speaker"].endswith(":0") else 0.0333
        with open(idx, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in ents)
        s = StreamingSegmentSampler(idx, "train", 1.0, licence_path="smoke")
        spk = torch.cat([s.batch(i, 8)["speaker"] for i in range(40)])
        share = float((spk == s.spk_index["librosa_example:0"]).float().mean())
        self.assertGreater(share, 0.75)


@unittest.skipUnless(HAVE_SMOKE, "smoke data not fetched")
class RealRunOnCpu(unittest.TestCase):
    def test_real_driver_end_to_end_gates_and_resume(self):
        import yaml
        import trainers.run_info as RI
        from trainers import real_run
        sys.path.insert(0, os.path.join(TRAINING, "scripts"))
        import train_speaker_encoder as TSE
        d = tempfile.mkdtemp()
        try:
            idx, _, _ = write_feature_fixture(d)
            cfg = yaml.safe_load(open(os.path.join(TRAINING, "configs", "smoke.yaml")))
            cfg["data"].update({"train_index": idx, "valid_index": idx, "segment_seconds": 1.0,
                                "speaker_encoders_train": [{"name": "ecapa_a", "arch": "ecapa"}]})
            cfg["stages"] = [{"name": "content_distillation", "steps": 4}, {"name": "reconstruction", "steps": 2},
                             {"name": "anonymization", "steps": 2}, {"name": "qat_int8", "steps": 2}]
            cfg["val_every"] = 2
            cfg["stage_gates"] = {"content_distillation": {"usage": {"metric": "code_usage_frac", "min": 0.0}},
                                  "reconstruction": {"wer": {"metric": "wer_recon_rel", "max": 0.10}}}
            cp = os.path.join(d, "cfg.yaml")
            yaml.safe_dump(cfg, open(cp, "w"))
            prof = yaml.safe_load(open(os.path.join(TRAINING, "configs", "gpu", "t4_16gb.yaml")))
            prof["batch_size"]["value"], prof["num_workers"]["value"], prof["checkpoint_every_steps"]["value"] = 2, 0, 2
            pp = os.path.join(d, "prof.yaml")
            yaml.safe_dump(prof, open(pp, "w"))
            md = os.path.join(d, "models_train")
            os.makedirs(md)
            enc = TSE.Encoder("ecapa", channels=16, dim=64).eval()     # untrained real ECAPA graph (plumbing only)
            torch.jit.trace(enc, torch.randn(1, 50, 80)).save(os.path.join(md, "ecapa_a.pt"))
            meta = {"name": "ecapa_a", "arch": "ecapa", "role": "TRAIN"}
            json.dump(meta, open(os.path.join(md, "ecapa_a.json"), "w"))
            ready = json.load(open(os.path.join(TRAINING, "readiness.json")))
            ready = dict(ready, verdict="GO", conditions={k: True for k in ready["conditions"]})
            rp = os.path.join(d, "ready.json")
            json.dump(ready, open(rp, "w"))
            out = os.path.join(d, "run")
            # refused while the real readiness says NO-GO
            with self.assertRaises(real_run.NotReady):
                real_run.run(cp, out, pp, "smoke", device="cpu", require_gpu=False, models_dir=md)
            orig = RI.git_state
            RI.git_state = lambda root: ("0" * 40, False)       # the test tree may be dirty; real runs refuse that
            try:
                reps = real_run.run(cp, out, pp, "smoke", readiness=rp, device="cpu", require_gpu=False, models_dir=md)
                self.assertTrue(reps["content_distillation"]["passed"])
                self.assertFalse(reps["reconstruction"]["passed"])            # WER gate NOT_MEASURED -> not passed
                self.assertEqual(reps["reconstruction"]["exit_criteria"]["wer"]["status"], "NOT_MEASURED")
                self.assertNotIn("anonymization", reps)                       # gated: never started
                side = json.load(open(os.path.join(out, "content_distillation", "last.pt.json")))
                self.assertEqual(side["git_sha"], "0" * 40)
                self.assertEqual(side["precision"], "fp32")                   # CPU stays fp32 even with a T4 profile
                reps2 = real_run.run(cp, out, pp, "smoke", readiness=rp, device="cpu", require_gpu=False,
                                     models_dir=md, resume=True, stages=["content_distillation"])
                self.assertTrue(reps2["content_distillation"]["passed"])
            finally:
                RI.git_state = orig
        finally:
            shutil.rmtree(d)


def mls_fixture(root, n_spk=6, books=2, utts=2, sec=1.0):
    for split, spks in (("train", range(100, 100 + n_spk)), ("dev", range(200, 202)), ("test", range(300, 302))):
        d = os.path.join(root, split)
        os.makedirs(d, exist_ok=True)
        tr, sg = [], []
        for s in spks:
            for b in range(books):
                for u in range(utts):
                    key = f"{s}_{1000 + b}_{u:06d}"
                    p = os.path.join(d, "audio", str(s), str(1000 + b), key + ".flac")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    sf.write(p, voiced(sec, 120 + s % 50), SR)
                    tr.append(f"{key}\thello world {u}")
                    sg.append(f"{key}\thttp://example.invalid/x\t0.0\t{sec + (s % 3)}")
        open(os.path.join(d, "transcripts.txt"), "w").write("\n".join(tr) + "\n")
        open(os.path.join(d, "segments.txt"), "w").write("\n".join(sg) + "\n")


class Mls(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_verify_requires_licence_and_records_provenance(self):
        from datasets import mls
        arc = os.path.join(self.d, "mls_english.tar.gz")
        with tarfile.open(arc, "w:gz") as t:
            t.add(__file__, arcname="x.py")
        prov = os.path.join(self.d, "prov.json")
        with self.assertRaises(mls.MlsError):
            mls.verify(arc, None, provenance_path=prov)
        bad = os.path.join(self.d, "lic_bad.txt")
        open(bad, "w").write("All rights reserved.")
        with self.assertRaises(mls.MlsError):
            mls.verify(arc, bad, provenance_path=prov)
        good = os.path.join(self.d, "lic.txt")
        open(good, "w").write("License: Creative Commons Attribution 4.0 International (CC BY 4.0)")
        with self.assertRaises(mls.MlsError):
            mls.verify(arc, good, expected_md5="0" * 32, provenance_path=prov)
        rec = mls.verify(arc, good, provenance_path=prov)
        self.assertEqual(len(rec["archive_sha256"]), 64)
        self.assertIn("mls_en", json.load(open(prov))["verified"])
        with self.assertRaises(mls.MlsError):
            mls.download("http://insecure", os.path.join(self.d, "a"))      # no guessed / non-https URL

    def test_manifest_subset_splits_and_gate(self):
        from datasets import gate as G
        from datasets import mls
        root = os.path.join(self.d, "mls_english")
        mls_fixture(root)
        rows = mls.build_rows(root)
        self.assertTrue(all(r["speaker"].startswith("mls_en:") for r in rows))
        sub = mls.speaker_balanced_subset(rows, 0.5)
        tr_h = sum(r["duration"] for r in sub if r["mls_split"] == "train")
        all_h = sum(r["duration"] for r in rows if r["mls_split"] == "train")
        self.assertLess(tr_h, all_h)
        self.assertEqual({r["speaker"] for r in sub if r["mls_split"] == "train"},
                         {r["speaker"] for r in rows if r["mls_split"] == "train"})        # all speakers kept
        with self.assertRaises(mls.MlsError):                       # commercial path: still COMMERCIAL_PENDING
            mls.make_manifests(root, os.path.join(self.d, "out"), 1.0)
        old = G.PROVENANCE
        G.PROVENANCE = os.path.join(self.d, "prov.json")
        json.dump({"verified": {"mls_en": {"archive_sha256": "a" * 64, "licence_text_sha256": "b" * 64}}}, open(G.PROVENANCE, "w"))
        try:
            idx = mls.make_manifests(root, os.path.join(self.d, "out"), 1.0)
            from datasets.build_manifests import verify
            verify(os.path.join(self.d, "out"))
            test = [json.loads(l) for l in open(os.path.join(self.d, "out", "test.jsonl"))]
            self.assertTrue(test and {r["role"] for r in test} == {"enroll", "trial"})
            self.assertIn("train", idx["manifests"])
        finally:
            G.PROVENANCE = old


def consent(cid, pseud, commercial=True, withdrawn=False):
    return {"consent_id": cid, "speaker_pseudonym": pseud, "form_version": "v1", "form_sha256": "0" * 64,
            "consent_timestamp": "2026-10-01T10:00:00Z", "adult_confirmed": True, "language": ["ar"],
            "dialect_primary": "Jordanian",
            "scopes": {"ml_training": True, "voice_anonymization_rnd": True, "commercial_product_development": commercial,
                       "model_evaluation": True, "derivative_model_training": True, "audio_redistribution": False},
            "withdrawal": {"status": "withdrawn" if withdrawn else "active", "withdrawn_at": None},
            "retention_until": "2031-01-01", "withdrawal_token_sha256": "a" * 64, "collector": "campaign-1"}


def session(spk, cid, sid, sr=48000):
    return {"speaker_id": spk, "consent_id": cid, "session_id": sid, "language": "ar", "dialect": "Jordanian",
            "recording_device": "phone_android", "environment": "quiet_room", "sampling_rate_hz": sr,
            "recorded_at": "2026-10-02", "duration_s": 10.0, "consent_scope_snapshot": "b" * 64, "withdrawal_status": "active"}


class ArabicImport(unittest.TestCase):
    def test_importer_validates_everything(self):
        from datasets import arabic_import as AI
        d = tempfile.mkdtemp()
        try:
            inc, store = os.path.join(d, "in"), os.path.join(d, "consents.jsonl")
            recs = []
            for i in range(12):
                cid, spk = f"CNS-{i:012x}", f"own_recordings:spk-{i:010x}"
                recs.append(consent(cid, spk, commercial=(i != 1), withdrawn=(i == 2)))
                for k in range(2):
                    sid = f"ses-{i:04x}{k:04x}"
                    sd = os.path.join(inc, cid, sid)
                    os.makedirs(sd)
                    json.dump(session(spk, cid, sid), open(os.path.join(sd, "session.json"), "w"))
                    x = np.repeat(voiced(2.0, 100 + 9 * i + 3 * k, i * 7 + k), 3)          # 2 s at 48 kHz
                    sf.write(os.path.join(sd, "a.wav"), np.stack([x, x], 1), 48000)    # stereo 48 kHz -> mono 16 kHz
                    open(os.path.join(sd, "a.txt"), "w", encoding="utf-8").write("مَرحبا كيف حالك اليوم يا صديقي")
            recs.append(consent("CNS-ffffffffffff", "own_recordings:spk-ffffffffff"))
            with open(store, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            bad = os.path.join(inc, "CNS-000000000003", "ses-00030000")
            open(os.path.join(bad, "b.wav"), "wb").write(b"RIFF\x00\x00garbage")                   # corrupt
            shutil.copy(os.path.join(bad, "a.wav"), os.path.join(inc, "CNS-000000000004", "ses-00040000", "dup.wav"))
            shutil.copy(os.path.join(bad, "a.txt"), os.path.join(inc, "CNS-000000000004", "ses-00040000", "dup.txt"))
            sf.write(os.path.join(bad, "silent.wav"), np.zeros(SR * 2, np.float32), SR)
            open(os.path.join(bad, "silent.txt"), "w").write("نص")
            sf.write(os.path.join(bad, "notext.wav"), voiced(2.0, 333, 5), SR)
            os.makedirs(os.path.join(inc, "CNS-0000000000aa", "ses-aaaaaaaa"))                       # no consent
            rep = AI.run(inc, store, os.path.join(d, "out"), "own-ar-test", "commercial", fractions=(0.6, 0.2, 0.2))
            self.assertIn("CNS-0000000000aa", rep["rejected"]["no consent record"])
            self.assertIn("CNS-000000000002", rep["rejected"]["consent withdrawn"])
            self.assertIn("corrupt", rep["rejected"])
            self.assertIn("silent", rep["rejected"])
            self.assertTrue(any("missing transcript" in v for v in rep["rejected"]["transcript"]))
            self.assertEqual([x["type"] for x in rep["duplicates"]], ["exact"])
            self.assertIn("own_recordings:spk-0000000001", rep["not_commercial_eligible_speakers"])
            self.assertEqual(rep["accepted"], 22)        # 11 active consented speakers x 2 sessions; extras all rejected
            out = [json.loads(l) for l in open(os.path.join(d, "out", "recordings.jsonl"), encoding="utf-8")]
            info = sf.info(os.path.join(d, "out", "audio", "spk-0000000000", "ses-00000000",
                                        next(r for r in out if r["speaker_id"].endswith("0000000000"))["recording_id"] + ".flac"))
            self.assertEqual((info.samplerate, info.channels), (16000, 1))
            from datasets.build_manifests import verify
            verify(os.path.join(d, "out", "manifests"))
            rows = [json.loads(l) for s in ("train", "valid") for l in open(os.path.join(d, "out", "manifests", f"{s}.jsonl"))]
            self.assertNotIn("own_recordings:spk-0000000001", {r["speaker"] for r in rows})       # opted out of commercial
            for r in out:
                self.assertTrue(r["transcript_normalized"] and "َ" not in r["transcript_normalized"])
        finally:
            shutil.rmtree(d)


class TeacherUnits(unittest.TestCase):
    def test_units_from_tiny_random_whisper(self):
        """Mechanics only: a tiny RANDOM Whisper (no download) -> units aligned to 20 ms, idempotent."""
        from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
        sys.path.insert(0, os.path.join(TRAINING, "scripts"))
        import compute_teacher_units as CTU
        d = tempfile.mkdtemp()
        try:
            cfg = WhisperConfig(d_model=32, encoder_layers=2, decoder_layers=1, encoder_attention_heads=2,
                                decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80,
                                max_source_positions=1500, max_target_positions=32, vocab_size=64,
                                pad_token_id=1, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
            tdir = os.path.join(d, "tiny-whisper")
            WhisperModel(cfg).save_pretrained(tdir)
            WhisperFeatureExtractor(feature_size=80).save_pretrained(tdir)
            rows = []
            for s in range(3):
                for u in range(2):
                    p = os.path.join(d, f"{s}_{u}.wav")
                    sf.write(p, voiced(1.5, 110 + 25 * s, s * 3 + u), SR)
                    rows.append({"path": p, "speaker": f"vctk:p{s}", "session": "1", "corpus": "vctk", "language": "en",
                                 "duration": 1.5, "sr": SR, "split": "train"})
            rep = CTU.run(rows, tdir, layer=1, k=8, out=os.path.join(d, "units"))
            from datasets.feature_cache import row_key
            u = np.load(os.path.join(d, "units", row_key(rows[0]) + ".npy"))
            self.assertEqual(len(u), 75)                     # 1.5 s at 50 units/s
            self.assertTrue(u.max() < 8)
            self.assertEqual(rep["new"], 6)
            self.assertEqual(CTU.run(rows, tdir, layer=1, k=8, out=os.path.join(d, "units"))["new"], 0)   # idempotent
        finally:
            shutil.rmtree(d)


class SpeakerEncoderTraining(unittest.TestCase):
    def test_train_export_roles_disjoint(self):
        sys.path.insert(0, os.path.join(TRAINING, "scripts"))
        import train_speaker_encoder as TSE
        d = tempfile.mkdtemp()
        try:
            idx, _, _ = write_feature_fixture(d, speakers=8)
            m_tr = TSE.train(idx, "enc_t", "TRAIN", 4, os.path.join(d, "r1"), os.path.join(d, "mt"), channels=32, dim=16,
                             batch=4, seg=1.0, licence_path="smoke", ckpt_every=2, device="cpu")
            m_va = TSE.train(idx, "enc_v", "VALID", 4, os.path.join(d, "r2"), os.path.join(d, "mt"), channels=32, dim=16,
                             batch=4, seg=1.0, licence_path="smoke", ckpt_every=2, device="cpu")
            self.assertNotEqual(m_tr["speaker_subset_sha256"], m_va["speaker_subset_sha256"])
            enc = torch.jit.load(os.path.join(d, "mt", "enc_t.pt"))
            e = enc(torch.randn(2, 120, 80))
            self.assertEqual(tuple(e.shape), (2, 16))
            self.assertTrue(torch.allclose(e.norm(dim=-1), torch.ones(2), atol=1e-4))
            again = TSE.train(idx, "enc_t", "TRAIN", 4, os.path.join(d, "r1"), os.path.join(d, "mt"), channels=32, dim=16,
                              batch=4, seg=1.0, licence_path="smoke", ckpt_every=2, device="cpu")   # resumes at step 4
            self.assertEqual(again["steps"], 4)
        finally:
            shutil.rmtree(d)


class NearDuplicate(unittest.TestCase):
    def test_reencoded_copy_detected_other_recording_not(self):
        from datasets import arabic_import as AI
        y = voiced(3.0, 130, 1)
        copy = np.clip(0.8 * y + 0.002 * np.random.default_rng(9).standard_normal(len(y)), -1, 1).astype(np.float32)
        other = voiced(3.0, 130, 2)
        self.assertTrue(AI.same_audio(AI.fingerprint(y), AI.fingerprint(copy)))
        self.assertFalse(AI.same_audio(AI.fingerprint(y), AI.fingerprint(other)))


class Mix(unittest.TestCase):
    def test_mix_enforces_disjoint_speakers_and_weights(self):
        from datasets import build_manifests as BM
        from datasets import mix as MX
        d = tempfile.mkdtemp()
        try:
            def src(name, corpus, lang, spks):
                rows = []
                for i, s in enumerate(spks):
                    for sess in ("1", "2"):
                        rows.append({"path": f"/x/{name}/{s}/{sess}.flac", "speaker": f"{corpus}:{s}", "session": sess,
                                     "corpus": corpus, "language": lang, "duration": 3.0, "sr": 16000,
                                     "split": "train" if i < len(spks) - 1 else "test",
                                     **({"role": "enroll" if sess == "1" else "trial"} if i == len(spks) - 1 else {})})
                BM.write(rows, os.path.join(d, name), 1, {}, {})
            src("en", "vctk", "en", ["p1", "p2", "p3", "p4"])
            src("ar", "vctk", "ar", ["q1", "q2", "q3"])          # stand-in corpus id; gate is mocked below
            cfg = {"licence_path": "commercial", "seed": 1, "language_weights": {"en": 0.7, "ar": 0.3},
                   "sources": [{"name": "en", "manifests": "en", "language": "en"}, {"name": "ar", "manifests": "ar", "language": "ar"}],
                   "out": "mix"}
            from datasets import gate as G
            old = G.PROVENANCE
            G.PROVENANCE = os.path.join(d, "prov.json")
            json.dump({"verified": {"vctk": {"archive_sha256": "a", "licence_text_sha256": "b"}}}, open(G.PROVENANCE, "w"))
            try:
                idx, n = MX.mix(cfg, base=d)
                tr = [json.loads(l) for l in open(os.path.join(d, "mix", "train.jsonl"))]
                en = sum(r["mix_weight"] for r in tr if r["language"] == "en")
                self.assertAlmostEqual(en, 0.7)
                json.dump([["vctk:p1", "vctk:p4"]], open(os.path.join(d, "same.json"), "w"))   # train p1 == test p4
                with self.assertRaises(MX.MixError):
                    MX.mix(dict(cfg, same_person="same.json", out="mix2"), base=d)
                json.dump({"verified": {}}, open(G.PROVENANCE, "w"))
                with self.assertRaises(MX.MixError):                                         # vctk now PENDING
                    MX.mix(dict(cfg, out="mix3"), base=d)
            finally:
                G.PROVENANCE = old
        finally:
            shutil.rmtree(d)


class Acceptance(unittest.TestCase):
    def good(self):
        M = lambda v, **k: dict({"value": v, "source": "MEASURED"}, **k)
        return {"informed_attacker_eer": M(0.31, ci_lower=0.26), "strong_pretrained_attacker_eer": M(0.41, ci_lower=0.33),
                "top1_identification": M(0.05, n_speakers=60), "matched_pseudo_eer": M(0.3, ci_lower=0.2),
                "multi_session_eer": M(0.3), "librispeech_wer_delta_abs": M(0.02), "spontaneous_wer_rel": M(0.1),
                "estoi": M(0.6), "human_mos": M(3.4, n_raters=25), "algorithmic_latency_ms": M(50.0),
                "android_rtf": M(0.3, measured_on_device=True, device="Pixel 8"), "int8_model_mb": M(7.6),
                "streaming_causal": M(True), "no_cloud_dependency": M(True)}

    def test_locked_and_strict(self):
        from evaluation import acceptance as A
        crit = A.load_criteria()
        self.assertEqual(crit["informed_attacker_eer"]["threshold"], 0.25)
        self.assertEqual(crit["android_rtf"]["threshold"], 0.5)
        self.assertEqual(A.evaluate(self.good())["verdict"], "PASS")
        r = self.good()
        r["informed_attacker_eer"]["value"] = 0.249
        ev = A.evaluate(r)
        self.assertEqual((ev["verdict"], ev["results"]["informed_attacker_eer"]["status"]), ("FAIL", "FAIL"))
        r = self.good()
        r["informed_attacker_eer"]["ci_lower"] = 0.19                     # point estimate fine, CI not
        self.assertEqual(A.evaluate(r)["results"]["informed_attacker_eer"]["status"], "FAIL")
        r = self.good()
        r["android_rtf"] = {"value": 0.2, "source": "MEASURED", "measured_on_device": False}
        self.assertEqual(A.evaluate(r)["results"]["android_rtf"]["status"], "INVALID")
        r = self.good()
        r["top1_identification"]["n_speakers"] = 20
        self.assertEqual(A.evaluate(r)["results"]["top1_identification"]["status"], "INVALID")
        r = self.good()
        r["human_mos"] = {"value": 3.5, "source": "ESTIMATED"}
        self.assertEqual(A.evaluate(r)["verdict"], "INCOMPLETE")
        r = self.good()
        r["algorithmic_latency_ms"]["value"] = 50.1
        self.assertEqual(A.evaluate(r)["verdict"], "FAIL")
        r = self.good()
        r["android_rtf"]["value"] = 0.5                                    # strictly below 0.5
        self.assertEqual(A.evaluate(r)["results"]["android_rtf"]["status"], "FAIL")

    def test_tampering_detected(self):
        from evaluation import acceptance as A
        d = tempfile.mkdtemp()
        try:
            c = json.load(open(A.CRITERIA))
            c["criteria"]["informed_attacker_eer"]["threshold"] = 0.15
            p = os.path.join(d, "c.json")
            json.dump(c, open(p, "w"))
            with self.assertRaises(A.CriteriaTampered):
                A.load_criteria(p)
        finally:
            shutil.rmtree(d)


class AndroidBenchmark(unittest.TestCase):
    def test_report_never_claims_unmeasured(self):
        import android_benchmark as AB
        rows = AB.build_report(None, None)
        self.assertTrue(all(r["status"] == "NOT_MEASURED" for r in rows.values()))
        emu = {"device": {"is_emulator": True}, "measured_on_device": False, "rtf_p50": 0.1, "call_ms_p50": 2.0}
        rows = AB.build_report(emu)
        self.assertEqual(rows["rtf_p50"]["status"], "EMULATOR")
        real = {"device": {"is_emulator": False, "model": "X"}, "measured_on_device": True, "rtf_p50": 0.1,
                "call_ms_p50": 2.0, "call_ms_p95": 2.5, "call_ms_p99": 3.0, "pss_mb_max": 40.0}
        rows = AB.build_report(real)
        self.assertEqual(rows["call_ms_p99"]["status"], "MEASURED")
        self.assertEqual(rows["dropped_callbacks"]["status"], "NOT_MEASURED")      # needs app integration
        self.assertEqual(rows["thermal_before"]["status"], "NOT_MEASURED")
        self.assertIn("NOT_MEASURED", AB.to_markdown(rows, real))


if __name__ == "__main__":
    unittest.main(verbosity=2)
