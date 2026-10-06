#!/usr/bin/env python3
"""B4 data-pipeline tests: augmentation, CTC targets, manifest builder (speaker-/session-
disjoint splits, leakage fail-fast, reproducible hashes). Synthetic fixture corpora only;
no dataset is downloaded.

  python3 training/tests/test_data_pipeline.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)

from datasets import augment as A  # noqa: E402
from datasets import build_manifests as BM  # noqa: E402
from datasets import ctc_targets as C  # noqa: E402

SR = 16000


def tone(sec=1.0, f=220.0, seed=0):
    t = np.arange(int(sec * SR)) / SR
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * f * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


class Augmentation(unittest.TestCase):
    def test_snr_is_exact(self):
        rng = np.random.default_rng(0)
        x = tone()
        n = np.random.default_rng(1).standard_normal(SR // 3).astype(np.float32)
        for snr in (0.0, 10.0, 25.0):
            y = A.add_noise(x, n, snr, rng)
            measured = 10 * np.log10(A.power(x) / A.power(y - x))
            self.assertAlmostEqual(measured, snr, delta=0.05)

    def test_reverb_keeps_length_power_and_alignment(self):
        x = np.zeros(SR, np.float32)
        x[1000] = 1.0
        rir = np.zeros(800)
        rir[50] = 1.0
        rir[300] = 0.5
        y = A.reverb(x, rir)
        self.assertEqual(len(y), len(x))
        self.assertEqual(int(np.argmax(np.abs(y))), 1000)     # direct path aligned
        self.assertAlmostEqual(A.power(y), A.power(x), delta=1e-6)

    def test_speed_changes_length_and_targets_follow(self):
        x = tone(1.0)
        for f in (0.9, 1.1):
            y = A.speed(x, f)
            self.assertAlmostEqual(len(y) / len(x), 1 / f, delta=0.01)
            units = np.arange(100)
            u = A.scale_targets(units, f)
            self.assertEqual(len(u), int(round(100 / f)))
            self.assertTrue(np.all(np.diff(u) >= 0))

    def test_pitch_and_speed_refused_outside_stage1(self):
        with self.assertRaises(ValueError):
            A.Augmenter(A.AugmentConfig(p_pitch=0.5), stage="reconstruction")
        with self.assertRaises(ValueError):
            A.Augmenter(A.AugmentConfig(p_speed=0.5), stage="anonymization")
        A.Augmenter(A.AugmentConfig(p_pitch=0.5, p_speed=0.5), stage="content_distillation")

    def test_codec_and_augmenter_finite_bounded_reproducible(self):
        x = tone(0.5)
        y = A.codec(x)
        self.assertTrue(np.all(np.isfinite(y)))
        cfg = A.AugmentConfig(p_noise=1.0, p_reverb=1.0, p_codec=1.0)
        aug = A.Augmenter(cfg, [np.random.default_rng(3).standard_normal(SR).astype(np.float32)],
                          [np.exp(-np.arange(400) / 80.0)])
        y1, r1 = aug(x, np.random.default_rng(42))
        y2, r2 = aug(x, np.random.default_rng(42))
        self.assertEqual(r1, r2)
        np.testing.assert_array_equal(y1, y2)
        self.assertLessEqual(float(np.abs(y1).max()), 1.0)
        self.assertTrue({"rir", "noise", "snr_db", "codec", "gain_db"} <= set(r1))
        self.assertTrue(cfg.snr_min <= r1["snr_db"] <= cfg.snr_max)


class CtcTargets(unittest.TestCase):
    def test_vocab_matches_config(self):
        import yaml
        for name in ("stream_anon_s.yaml", "stream_anon_m.yaml"):
            cfg = yaml.safe_load(open(os.path.join(TRAINING, "configs", name)))
            self.assertEqual(cfg["model"]["n_phones"], C.n_tokens(), name)
        self.assertEqual(C.VOCAB[0], "<blank>")
        self.assertEqual(len(set(C.VOCAB)), len(C.VOCAB))

    def test_english_roundtrip(self):
        ids = C.encode("Hello, World! It's 42.", "en")
        self.assertNotIn(0, ids)
        self.assertEqual(C.decode(ids), "hello world it's 42")

    def test_arabic_normalisation(self):
        # diacritics + tatweel removed, alef forms unified, ta marbuta/alef maqsura mapped, digits
        self.assertEqual(C.normalize("مَدْرَسـَة", "ar"), "مدرسه")
        self.assertEqual(C.normalize("أحمد إلى آخر", "ar"), "احمد الي اخر")
        self.assertEqual(C.normalize("٣٤ ۵", "ar"), "34 5")
        self.assertEqual(C.decode(C.encode("الكِتابُ", "ar")), "الكتاب")
        for ch in "ةىآأإ":
            self.assertNotIn(ch, C.VOCAB)


def write_wav(path, sec, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, tone(sec, 150 + 10 * (seed % 20), seed), SR)


def fixture(root, n_spk=40, sessions=3, utts=2, subset="train-clean-100", corpus_dir="LibriTTS_R", first=100):
    """LibriSpeech-layout corpus: <root>/<subset>/<spk>/<chapter>/<spk>-<chapter>-<n>.flac"""
    base = os.path.join(root, corpus_dir)
    for s in range(first, first + n_spk):
        for c in range(sessions):
            d = os.path.join(base, subset, str(s), str(1000 + c))
            lines = []
            for u in range(utts):
                key = f"{s}-{1000 + c}-{u:04d}"
                write_wav(os.path.join(d, key + ".flac"), 0.3, s * 10 + c)
                lines.append(f"{key} HELLO WORLD {u}")
            open(os.path.join(d, f"{s}-{1000 + c}.trans.txt"), "w").write("\n".join(lines) + "\n")
    return base


def consent_record(cid, pseudonym, commercial=True, withdrawn=False, evaluation=True):
    return {"consent_id": cid, "speaker_pseudonym": pseudonym, "form_version": "consent-ar-en-v1.0", "form_sha256": "0" * 64,
            "consent_timestamp": "2026-10-01T10:00:00Z", "adult_confirmed": True, "language": ["ar"], "dialect_primary": "Jordanian",
            "scopes": {"ml_training": True, "voice_anonymization_rnd": True, "commercial_product_development": commercial,
                       "model_evaluation": evaluation, "derivative_model_training": commercial, "audio_redistribution": False},
            "withdrawal": {"status": "withdrawn" if withdrawn else "active", "withdrawn_at": None}, "retention_until": "2031-10-01",
            "withdrawal_token_sha256": "a" * 64, "collector": "campaign-2026-amman-1"}


class Consent(unittest.TestCase):
    def test_schema_valid_record_and_rejections(self):
        from datasets import consent as CO
        rec = consent_record("CNS-0123456789ab", "own_recordings:spk-0123456789")
        self.assertEqual(CO.validate_record(rec), [])
        self.assertTrue(CO.validate_record(dict(rec, name="Ahmad")))              # no extra personal fields
        self.assertTrue(CO.validate_record(dict(rec, adult_confirmed=False)))
        self.assertTrue(CO.validate_record(dict(rec, notes="call me 0791234567")))
        self.assertTrue(CO.validate_record(dict(rec, speaker_pseudonym="own_recordings:ahmad")))
        bad = dict(rec, scopes=dict(rec["scopes"], ml_training=False))
        self.assertTrue(any("requires ml_training" in e for e in CO.validate_record(bad)))
        self.assertTrue(CO.validate_record(dict(rec, scopes=dict(rec["scopes"], sell_voice=True))))
        self.assertEqual(CO.commercial_eligibility(rec), "COMMERCIAL_TRAINING_ELIGIBLE")
        for k in CO.COMMERCIAL_SCOPES:   # every one of the five scopes is necessary
            self.assertEqual(CO.commercial_eligibility(dict(rec, scopes=dict(rec["scopes"], **{k: False}))),
                             "NOT_COMMERCIAL_TRAINING_ELIGIBLE", k)
        self.assertEqual(CO.commercial_eligibility(dict(rec, withdrawal={"status": "withdrawn", "withdrawn_at": None})),
                         "NOT_COMMERCIAL_TRAINING_ELIGIBLE")

    def test_session_schema(self):
        from datasets import consent as CO
        ses = {"speaker_id": "own_recordings:spk-0123456789", "consent_id": "CNS-0123456789ab", "session_id": "ses-0a1b2c3d",
               "language": "ar", "dialect": "Jordanian", "recording_device": "phone_android", "environment": "quiet_room",
               "sampling_rate_hz": 48000, "recorded_at": "2026-10-02", "duration_s": 1800.0, "age_bucket": "25-34",
               "consent_scope_snapshot": "b" * 64, "withdrawal_status": "active"}
        self.assertEqual(CO.validate_session(ses), [])
        self.assertTrue(CO.validate_session(dict(ses, device_serial="R58M12345")))   # no device identifiers
        self.assertTrue(CO.validate_session(dict(ses, recording_device="Samsung SM-A515F")))

    def test_check_rows(self):
        from datasets import consent as CO
        p = "own_recordings:spk-0123456789"
        row = {"corpus": "own_recordings", "speaker": p, "consent_id": "CNS-0123456789ab", "split": "train"}
        ok = {"CNS-0123456789ab": consent_record("CNS-0123456789ab", p)}
        self.assertEqual(CO.check_rows([row], ok, "commercial"), [])
        self.assertTrue(CO.check_rows([dict(row, consent_id="CNS-ffffffffffff")], ok))
        nc = {"CNS-0123456789ab": consent_record("CNS-0123456789ab", p, commercial=False)}
        self.assertTrue(CO.check_rows([row], nc, "commercial"))
        self.assertEqual(CO.check_rows([row], nc, "research"), [])
        wd = {"CNS-0123456789ab": consent_record("CNS-0123456789ab", p, withdrawn=True)}
        self.assertTrue(CO.check_rows([row], wd, "research"))
        other = {"CNS-0123456789ab": consent_record("CNS-0123456789ab", "own_recordings:spk-aaaaaaaaaa")}
        self.assertTrue(CO.check_rows([row], other, "research"))
        ne = {"CNS-0123456789ab": consent_record("CNS-0123456789ab", p, evaluation=False)}
        self.assertTrue(CO.check_rows([dict(row, split="test")], ne, "research"))
        old = {"CNS-0123456789ab": dict(consent_record("CNS-0123456789ab", p), retention_until="2020-01-01")}
        self.assertTrue(CO.check_rows([row], old, "research"))


class ManifestBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.libri = fixture(cls.tmp)
        cls.attack = fixture(cls.tmp, n_spk=6, sessions=1, subset="train-clean-360", corpus_dir="LibriSpeech", first=500)
        g = os.path.join(cls.tmp, "own")
        cls.consents = {}
        for s in range(4):
            spk = f"spk-{s:010x}"
            for sess in ("day1", "day2"):
                write_wav(os.path.join(g, spk, sess, "a.wav"), 0.3, 900 + s)
            cid = f"CNS-{s:012x}"
            open(os.path.join(g, spk, "consent_id.txt"), "w").write(cid)
            cls.consents[cid] = consent_record(cid, f"own_recordings:{spk}", commercial=(s % 2 == 0))
        cls.store = os.path.join(cls.tmp, "consents.jsonl")
        with open(cls.store, "w") as f:
            for rec in cls.consents.values():
                f.write(json.dumps(rec) + "\n")
        cls.generic = g
        cls.rows = (BM.build_librispeech(cls.libri, "mls_en", ["train-clean-100"])
                    + BM.build_librispeech(cls.attack, "librispeech", ["train-clean-360"])
                    + BM.build_generic(g, "own_recordings", "ar"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def split(self, seed=2026):
        return BM.assign_splits(self.rows, seed, fractions=(0.8, 0.1, 0.1))

    def test_speaker_disjoint_and_session_disjoint(self):
        rows, dropped = self.split()
        BM.assert_no_leakage(rows)
        by = {}
        for r in rows:
            by.setdefault(r["split"], set()).add(r["speaker"])
        splits = sorted(by)
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                self.assertFalse(by[splits[i]] & by[splits[j]])
        self.assertEqual({r["speaker"].split(":")[0] for r in rows if r["split"] == "attacker_train"}, {"librispeech"})
        test = [r for r in rows if r["split"] == "test"]
        self.assertTrue(test)
        for spk in {r["speaker"] for r in test}:
            enr = {r["session"] for r in test if r["speaker"] == spk and r["role"] == "enroll"}
            tri = {r["session"] for r in test if r["speaker"] == spk and r["role"] == "trial"}
            self.assertTrue(enr and tri and not (enr & tri))

    def test_injected_speaker_overlap_fails(self):
        rows, _ = self.split()
        t = next(r for r in rows if r["split"] == "test")
        bad = rows + [dict(t, split="train", role=None)]
        with self.assertRaises(BM.LeakageError):
            BM.assert_no_leakage(bad)

    def test_cross_corpus_same_speaker_space_overlap_fails(self):
        rows, _ = self.split()
        t = next(r for r in rows if r["split"] == "test" and r["corpus"] == "mls_en")
        sid = t["speaker"].split(":")[1]
        alias = dict(t, corpus="librispeech", speaker=f"librispeech:{sid}", subset="train-clean-100", split="train")
        alias.pop("role", None)
        with self.assertRaises(BM.LeakageError):
            BM.assert_no_leakage(rows + [alias])

    def test_session_overlap_between_enroll_and_trial_fails(self):
        rows, _ = self.split()
        t = next(r for r in rows if r["split"] == "test" and r["role"] == "trial")
        with self.assertRaises(BM.LeakageError):
            BM.assert_no_leakage(rows + [dict(t, role="enroll", path=t["path"] + "#dup")])

    def test_excluded_and_reserved_never_in_train(self):
        extra = dict(self.rows[0], speaker="librispeech:8312", corpus="librispeech")   # known VC target
        rows, dropped = BM.assign_splits(self.rows + [extra], 1)
        self.assertNotIn("librispeech:8312", {r["speaker"] for r in rows})
        self.assertEqual(dropped.get("excluded_speaker"), 1)
        bad = rows + [dict(extra, split="train")]
        with self.assertRaises(BM.LeakageError):
            BM.assert_no_leakage(bad)
        reserved = dict(self.rows[0], speaker="librispeech:77", corpus="librispeech", subset="test-clean", split="train")
        with self.assertRaises(BM.LeakageError):
            BM.assert_no_leakage(rows + [reserved])

    def test_attacker_pool_speaker_dropped_from_model_splits(self):
        a = next(r for r in self.rows if r["subset"] == "train-clean-360")
        same = dict(a, corpus="mls_en", speaker="mls_en:" + a["speaker"].split(":")[1], subset="train-clean-100",
                    session="77", path=a["path"] + "#x")
        rows, dropped = BM.assign_splits(self.rows + [same], 3)
        self.assertGreaterEqual(dropped.get("speaker_also_in_attacker_pool", 0), 1)
        BM.assert_no_leakage(rows)

    def test_licence_paths(self):
        rows, _ = self.split()
        self.assertTrue(BM.licence_check(rows, "commercial"))      # mls_en is COMMERCIAL_PENDING (not downloaded/verified)
        self.assertTrue(BM.licence_check(rows, "research"))        # own recordings need consent records
        self.assertEqual(BM.licence_check(rows, "research", self.consents), [])
        opted_out = [dict(r, split="train") for r in rows if r["corpus"] == "own_recordings"
                     and not self.consents[r["consent_id"]]["scopes"]["commercial_product_development"]]
        self.assertTrue(opted_out)
        self.assertTrue(any("NOT_COMMERCIAL_TRAINING_ELIGIBLE" in e for e in BM.licence_check(opted_out, "commercial", self.consents)))
        ok = [dict(r, corpus="vctk", speaker="vctk:" + r["speaker"].split(":")[1]) for r in rows]
        self.assertTrue(BM.licence_check(ok, "commercial"))                       # E1 licence but provenance pending
        prov = {"vctk": {"archive_sha256": "a" * 64, "licence_text_sha256": "b" * 64}}
        self.assertEqual(BM.licence_check(ok, "commercial", prov=prov), [])       # verified -> admitted
        unv = [dict(rows[0], corpus="masc", split="train")]
        self.assertTrue(BM.licence_check(unv, "research"))                        # LICENSE_UNVERIFIED: no path at all
        nc = [dict(rows[0], corpus="qasr", split="train")]
        self.assertTrue(BM.licence_check(nc, "commercial"))

    def test_deterministic_hashes_and_verify(self):
        o1, o2 = os.path.join(self.tmp, "m1"), os.path.join(self.tmp, "m2")
        for o in (o1, o2):
            rows, dropped = self.split(seed=7)
            BM.write(rows, o, 7, {"fixture": True}, dropped)
        i1 = json.load(open(os.path.join(o1, "manifest_index.json")))
        i2 = json.load(open(os.path.join(o2, "manifest_index.json")))
        self.assertEqual(i1["manifests"], i2["manifests"])
        rows_b, _ = self.split(seed=8)
        self.assertNotEqual(sorted((r["speaker"], r["split"]) for r in rows_b),
                            sorted((r["speaker"], r["split"]) for r in self.split(seed=7)[0]))
        BM.verify(o1)
        st = json.load(open(os.path.join(o1, "stats.json")))
        self.assertIn("train/ALL", st)
        self.assertTrue(open(os.path.join(o1, "stats.md")).read().startswith("| split/group"))
        # tamper: move one test row into train -> hash mismatch AND leakage
        p = os.path.join(o1, "train.jsonl")
        t = open(os.path.join(o1, "test.jsonl")).readline()
        r = json.loads(t)
        r["split"] = "train"
        open(p, "a").write(json.dumps(r) + "\n")
        with self.assertRaises(BM.LeakageError):
            BM.verify(o1)

    def test_cli_builds_and_verify_exits_1_on_leakage(self):
        out = os.path.join(self.tmp, "cli")
        script = os.path.join(TRAINING, "datasets", "build_manifests.py")
        r = subprocess.run([sys.executable, script, "--out", out, "--seed", "5",
                            "--librispeech", f"{self.libri}:mls_en:train-clean-100",
                            "--generic", f"{self.generic}:own_recordings:ar"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)          # default commercial path: unverified corpora refused
        self.assertIn("LICENCE", r.stderr)
        r = subprocess.run([sys.executable, script, "--out", out, "--seed", "5", "--licence-path", "research",
                            "--consent-store", self.store,
                            "--librispeech", f"{self.libri}:mls_en:train-clean-100",
                            "--generic", f"{self.generic}:own_recordings:ar"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sys.executable, script, "--verify", out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # inject the same speaker into train AND test, regenerate the index -> must still fail
        test_rows = [json.loads(l) for l in open(os.path.join(out, "test.jsonl"))]
        with open(os.path.join(out, "train.jsonl"), "a") as f:
            f.write(json.dumps(dict(test_rows[0], split="train")) + "\n")
        idx = json.load(open(os.path.join(out, "manifest_index.json")))
        import hashlib
        idx["manifests"]["train"] = hashlib.sha256(open(os.path.join(out, "train.jsonl"), "rb").read()).hexdigest()
        json.dump(idx, open(os.path.join(out, "manifest_index.json"), "w"))
        r = subprocess.run([sys.executable, script, "--verify", out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("LEAKAGE", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
