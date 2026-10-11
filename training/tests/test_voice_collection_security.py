"""Regression + synthetic tests for docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md (R7, R8, C1, C2, C10).

Synthetic data only: generated tones and fake records. No Telegram, no network, no real voices.
  python3 training/tests/test_voice_collection_security.py
"""
import copy
import datetime
import io
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from datasets import build_manifests as BM  # noqa: E402
from datasets import collection_core as CC  # noqa: E402
from datasets import consent as CO  # noqa: E402
from datasets import leakage  # noqa: E402

NOW = datetime.datetime(2026, 10, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)


def record(**kw):
    r = {"consent_id": "CNS-0123456789ab", "speaker_pseudonym": "own_recordings:spk-0123456789",
         "form_version": "consent-ar-en-v0.1", "form_sha256": "0" * 64, "consent_timestamp": "2026-10-01T10:00:00Z",
         "adult_confirmed": True, "language": ["ar"], "dialect_primary": "Jordanian",
         "scopes": {"ml_training": True, "voice_anonymization_rnd": True, "commercial_product_development": True,
                    "model_evaluation": True, "derivative_model_training": True, "audio_redistribution": False,
                    "listening_tests": False},
         "withdrawal": {"status": "active", "withdrawn_at": None}, "retention_until": "2028-10-01",
         "withdrawal_token_sha256": "a" * 64, "collector": "campaign-2026-amman-1"}
    r.update(kw)
    return r


def errs(rec):
    return CO.validate_record(rec, now=NOW)


def synthetic_ogg_bytes(seed=0, seconds=1.0, sr=16000):
    """A harmonic tone (no speech) encoded in memory; stands in for a voice note."""
    t = np.arange(int(sr * seconds)) / sr
    x = 0.1 * sum(np.sin(2 * np.pi * (140 + 10 * seed) * k * t) / k for k in range(1, 8))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="OGG", subtype="VORBIS")
    return buf.getvalue()


# ---------------------------------------------------------------- R7: retention / time formats
class RetentionAndTime(unittest.TestCase):
    def test_baseline_valid(self):
        self.assertEqual(errs(record()), [])

    def test_retention_never_and_non_dates_rejected(self):
        for v in ("never", "", "2028/10/01", "2028-10-1", "01-10-2028", "2028-10-01T00:00:00Z", "9999-12-31",
                  "2028-02-30", "2028-13-01", None, 20281001, "٢٠٢٨-١٠-٠١"):
            with self.subTest(v=v):
                self.assertTrue(errs(record(retention_until=v)))

    def test_retention_policy_window(self):
        self.assertEqual(errs(record(retention_until="2028-10-01")), [])              # exactly 24 months
        self.assertTrue(errs(record(retention_until="2028-10-02")))                   # one day over
        self.assertTrue(errs(record(retention_until="2026-10-01")))                   # not after consent
        self.assertTrue(errs(record(retention_until="2020-01-01")))                   # before consent

    def test_consent_timestamp_strict_utc(self):
        for v in ("yesterday", "2026-10-01", "2026-10-01T10:00:00", "2026-10-01T10:00:00+03:00",
                  "2026-10-01 10:00:00Z", "2026-02-30T10:00:00Z", "2026-10-01T25:00:00Z", "2026-10-01T10:00Z",
                  "2027-01-01T00:00:00Z", None, 1759312800):
            with self.subTest(v=v):
                self.assertTrue(errs(record(consent_timestamp=v)))
        self.assertEqual(errs(record(consent_timestamp="2026-10-01T10:00:00+00:00")), [])
        self.assertEqual(errs(record(consent_timestamp="2026-10-01T10:00:00.123Z")), [])

    def test_withdrawn_at_checked_when_present(self):
        self.assertEqual(errs(record(withdrawal={"status": "withdrawn", "withdrawn_at": "2026-10-05T08:00:00Z"})), [])
        for v in ("2026-09-30T00:00:00Z", "soon", "2026-10-05T08:00:00+01:00", "2030-01-01T00:00:00Z"):
            with self.subTest(v=v):
                self.assertTrue(errs(record(withdrawal={"status": "withdrawn", "withdrawn_at": v})))

    def test_session_recorded_at_strict(self):
        ses = {"speaker_id": "own_recordings:spk-0123456789", "consent_id": "CNS-0123456789ab", "session_id": "ses-0a1b2c3d",
               "language": "ar", "dialect": "Jordanian", "recording_device": "phone_android", "environment": "quiet_room",
               "sampling_rate_hz": 48000, "recorded_at": "2026-10-02", "duration_s": 30.0,
               "consent_scope_snapshot": "b" * 64, "withdrawal_status": "active"}
        self.assertEqual(CO.validate_session(ses), [])
        for v in ("2026-02-30", "02/10/2026", "never", None):
            with self.subTest(v=v):
                self.assertTrue(CO.validate_session(dict(ses, recorded_at=v)))

    def test_check_rows_validates_in_memory_records_and_never_expiry(self):
        p, cid = "own_recordings:spk-0123456789", "CNS-0123456789ab"
        row = {"corpus": "own_recordings", "speaker": p, "consent_id": cid, "split": "train"}
        self.assertEqual(CO.check_rows([row], {cid: record()}, "research", today="2027-01-01"), [])
        self.assertTrue(CO.check_rows([row], {cid: record(retention_until="never")}, "research", today="2027-01-01"))
        self.assertTrue(CO.check_rows([row], {cid: record()}, "research", today="2028-10-02"))   # expired
        with self.assertRaises(ValueError):
            CO.check_rows([row], {cid: record()}, "research", today="tomorrow")


# ---------------------------------------------------------------- contact data / Telegram ids
class ContactAndTelegramIds(unittest.TestCase):
    BAD = ["@some_user", "telegram @SomeUser", "＠some_user",                       # fullwidth @
           "@some​user_x", "@⁠some_user", "@ some_user",                 # zero-width / word joiner / space
           "t.me/some_user", "T.ME/joinchat/abc", "telegram.me/someone", "telegram.dog/x", "t . me / x",
           "https://t.me/+AbCdEf", "tg://user?id=123456789", "TG://resolve?domain=abc",
           "id: 123456789", "user id 987654321", "chat_id=1234567", "١٢٣٤٥٦٧٨٩", "۱۲۳۴۵۶۷۸۹",  # Arabic-Indic / Persian digits
           "１２３４５６７８９", "call 079 123 4567", "+962-79-123-4567", "0791234567",
           "name@example.com", "name＠example.com", "ab­@cd.com"]
    OK = ["quiet room, slight echo", "speaker prefers MSA", "session 2 of 3", "campaign-2026-amman-1", "noise level ok"]

    def test_notes_rejected(self):
        for v in self.BAD:
            with self.subTest(v=v):
                self.assertTrue(errs(record(notes=v)), v)

    def test_notes_allowed(self):
        for v in self.OK:
            with self.subTest(v=v):
                self.assertEqual(errs(record(notes=v)), [])

    def test_collector_form_version_dialect_secondary_scanned(self):
        self.assertTrue(errs(record(collector="@campaign_owner")))
        self.assertTrue(errs(record(collector="t.me/recruiter")))
        self.assertTrue(errs(record(collector="chat_id=1234567")))
        self.assertTrue(errs(record(collector="user-id 55555")))
        self.assertEqual(errs(record(collector="paid-campaign-12")), [])
        self.assertTrue(errs(record(form_version="v1 contact @someone_x")))
        self.assertTrue(errs(record(dialect_secondary=["Gulf", "tg://user?id=55555"])))
        self.assertEqual(errs(record(dialect_secondary=["Gulf"])), [])


# ---------------------------------------------------------------- R8: SVQ + trial reservation
class SvqAndTrialReservation(unittest.TestCase):
    base = {"path": "a.wav", "session": "s1", "language": "ar", "duration": 2.0, "sr": 16000}

    def row(self, corpus, subset, split, spk="x1"):
        return dict(self.base, speaker=f"{corpus}:{spk}", corpus=corpus, subset=subset, split=split)

    def test_leakage_check_blocks_svq_everywhere_but_test(self):
        for corpus in ("svq", "google_svq", "simple_voice_questions"):
            for split in ("train", "valid", "attacker_train"):
                with self.subTest(corpus=corpus, split=split):
                    self.assertTrue(any("reserved subset" in e for e in leakage.check([self.row(corpus, "clean", split)])))
        self.assertEqual(leakage.check([self.row("svq", "clean", "test")]), [])

    def test_trial_campaign_blocked(self):
        for split in ("train", "valid", "attacker_train"):
            self.assertTrue(leakage.check([self.row("own_recordings", "trial-2026-10", split, "spk-0123456789")]))
        self.assertEqual(leakage.check([self.row("own_recordings", "pilot-2026-11", "train", "spk-0123456789")]), [])

    def test_assign_splits_drops_reserved_and_keeps_attacker_pool(self):
        rows = [self.row("svq", "clean", "train", "a"), self.row("own_recordings", "trial-1", "train", "spk-00000000aa"),
                dict(self.base, speaker="librispeech:200", corpus="librispeech", subset="train-clean-360", split="x")]
        out, dropped = BM.assign_splits(rows, 1)
        self.assertEqual(dropped.get("reserved_subset"), 2)
        self.assertEqual([r["split"] for r in out], ["attacker_train"])

    def test_every_manifest_entry_point_refuses_svq(self):
        bad = [dict(self.row("svq", "clean", "train"), duration=2.0)]
        with self.assertRaises(BM.LeakageError):
            BM.assert_no_leakage(bad)                          # build_manifests main, mls, mix, arabic_import, verify
        tmp = tempfile.mkdtemp()
        try:
            with self.assertRaises(BM.LeakageError):
                BM.write(bad, os.path.join(tmp, "m"), 1, {}, {})   # the writer itself is gated
            self.assertFalse(os.path.exists(os.path.join(tmp, "m", "train.jsonl")))
        finally:
            shutil.rmtree(tmp)

    def test_frozen_eval_hash_and_criteria_unchanged(self):
        import hashlib
        root = os.path.dirname(TRAINING)
        def sha(p):
            with open(os.path.join(root, p), "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(sha("experiments/llvc_vs_dsp_48spk/manifest.json"),
                         "9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685")
        self.assertEqual(sha("data/acceptance_criteria.json"),
                         "58a893c21543d2075b4d962014e01cac093e9cf625af2a5b1b8a2fdd13af1aaf")


# ---------------------------------------------------------------- C1/C2/C10: synthetic journey
class SyntheticCollectionJourney(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fetches = []
        self.lineage = CC.LineageIndex()
        self.audio = {f"file-{i}": synthetic_ogg_bytes(i) for i in range(4)}

        def fetch(ref):
            self.fetches.append(ref)
            return self.audio[ref]
        self.core = CC.CollectionCore(fetch, CC.DirStore(self.tmp), self.lineage, "consent-ar-en-v0.1", "0" * 64,
                                      "campaign-synthetic-1", now=lambda: NOW)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def files(self):
        return sorted(os.path.relpath(os.path.join(d, f), self.tmp) for d, _, fs in os.walk(self.tmp) for f in fs)

    def consent(self, key, agree_all=True):
        self.core.start(key)
        self.core.confirm_adult(key)
        for s in CC.SCOPES:
            self.core.answer_scope(key, s, agree_all and s not in ("audio_redistribution",))
        return self.core.confirm(key)

    def test_no_fetch_no_storage_before_consent(self):
        k = "hmac-A"
        self.assertEqual(self.core.media(k, "voice", "file-0")[0], "REJECTED_NO_CONSENT")      # never started
        self.core.start(k)
        self.assertEqual(self.core.media(k, "voice", "file-0")[0], "REJECTED_NO_CONSENT")      # info screen
        self.core.confirm_adult(k)
        for s in CC.SCOPES[:3]:
            self.core.answer_scope(k, s, True)
        self.assertEqual(self.core.media(k, "voice", "file-0")[0], "REJECTED_NO_CONSENT")      # mid-flow
        with self.assertRaises(CC.GateError):
            self.core.confirm(k)                                                                # unanswered scopes
        self.assertEqual(self.fetches, [])
        self.assertEqual(self.files(), [])

    def test_age_gate(self):
        self.core.start("hmac-B")
        with self.assertRaises(CC.GateError):
            self.core.answer_scope("hmac-B", "ml_training", True)
        with self.assertRaises(CC.GateError):
            self.core.confirm("hmac-B")

    def test_consent_record_valid_and_token_not_stored_in_plain(self):
        cid, token = self.consent("hmac-C")
        rec = self.core.records[cid]
        self.assertEqual(CO.validate_record(rec, now=NOW), [])
        self.assertNotIn(token, repr(self.core.records) + repr(self.core.state))
        self.assertEqual(rec["retention_until"], "2028-10-11")

    def test_rejected_kinds_and_contexts_never_fetched(self):
        k = "hmac-D"
        self.consent(k)
        self.core.open_prompt(k, "p1")
        for kind, kw in (("audio", {}), ("video_note", {}), ("document", {}), ("voice", {"forwarded": True}),
                         ("voice", {"edited": True}), ("voice", {"chat_type": "group"}),
                         ("voice", {"chat_type": "channel"}), ("voice", {"prompt_id": "other"})):
            with self.subTest(kind=kind, kw=kw):
                self.assertNotEqual(self.core.media(k, kind, "file-0", **{"prompt_id": "p1", **kw})[0], "STORED")
        self.assertEqual(self.fetches, [])
        self.assertEqual(self.files(), [])

    def test_voice_without_open_prompt_not_fetched(self):
        k = "hmac-E"
        self.consent(k)
        self.assertEqual(self.core.media(k, "voice", "file-0", prompt_id="p1")[0], "REJECTED_NO_PROMPT")
        self.assertEqual(self.fetches, [])

    def test_store_then_withdraw_deletes_all_registered_artefacts(self):
        k = "hmac-F"
        cid, token = self.consent(k)
        self.core.open_prompt(k, "p1")
        d, path = self.core.media(k, "voice", "file-1", prompt_id="p1")
        self.assertEqual(d, "STORED")
        with open(path, "rb") as f:
            self.assertEqual(f.read(), self.audio["file-1"])
        pseud = self.core.records[cid]["speaker_pseudonym"]
        feat = os.path.join(self.tmp, "derived_features.npy")             # a synthetic derived artefact
        np.save(feat, np.zeros(4))
        self.lineage.record(pseud, feat, "features")
        report = self.core.withdraw(token=token)
        self.assertEqual(self.files(), [])
        self.assertEqual({k for k, _ in report["deleted"]}, {"raw_audio", "features"})
        self.assertEqual(report["not_deletable"], list(CC.NOT_DELETABLE))
        self.assertEqual(self.core.records[cid]["withdrawal"]["status"], "withdrawn")
        self.assertEqual(CO.validate_record(self.core.records[cid], now=NOW), [])
        row = {"corpus": "own_recordings", "speaker": pseud, "consent_id": cid, "split": "train"}
        self.assertTrue(CO.check_rows([row], self.core.records, "research", today="2026-10-12"))
        self.assertEqual(self.core.media(k, "voice", "file-2", prompt_id="p1")[0], "REJECTED_NO_CONSENT")

    def test_withdrawal_during_fetch_discards(self):
        k = "hmac-G"
        cid, token = self.consent(k)
        self.core.open_prompt(k, "p1")
        inner = self.core.fetch

        def racing_fetch(ref):
            data = inner(ref)
            self.core.withdraw(key=k)                 # participant withdraws while the file downloads
            return data
        self.core.fetch = racing_fetch
        self.assertEqual(self.core.media(k, "voice", "file-3", prompt_id="p1")[0], "DISCARDED_WITHDRAWN")
        self.assertEqual(self.files(), [])

    def test_unknown_token_and_scope_rejected(self):
        with self.assertRaises(CC.GateError):
            self.core.withdraw(token="not-a-token")
        self.core.start("hmac-H")
        self.core.confirm_adult("hmac-H")
        with self.assertRaises(CC.GateError):
            self.core.answer_scope("hmac-H", "sell_voice", True)
        with self.assertRaises(CC.GateError):
            self.core.answer_scope("hmac-H", "ml_training", "yes")


if __name__ == "__main__":
    unittest.main(verbosity=1)
