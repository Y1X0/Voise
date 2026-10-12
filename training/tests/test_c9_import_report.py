"""C9 / R10 regression tests for the Arabic importer's outputs (datasets/arabic_import.py).

import_report.json sits next to the manifests. It used to echo source file names, unvalidated
directory names and quoted validation values, so a dated source name such as
"voice_2026-10-02_14-03-22.wav" reached the dataset directory. These tests pin the fix
(positional references, scrubbed messages) and its functional compatibility.

Needs scipy + librosa (importer resampling and fingerprinting): runs in the CI job that installs
them. Synthetic only: generated harmonic tones, fake ids, temporary directories.
"""
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from datasets import arabic_import as AI  # noqa: E402
from datasets import build_manifests as BM  # noqa: E402
from datasets import consent as CO  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")

SR = 16000
SOURCE_HINTS = ("voice_", "audio_", "dup_17", "evening", "batch", ".wav")


def voiced(sec, f0, seed):
    t = np.arange(int(sec * SR)) / SR
    rng = np.random.default_rng(seed)
    am = np.interp(t, np.linspace(0, sec, 12), rng.uniform(0.1, 1.0, 12))
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * am
    return (0.2 * x / np.abs(x).max() + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def consent(cid, spk):
    return {"consent_id": cid, "speaker_pseudonym": spk, "form_version": "consent-ar-en-v1.0", "form_sha256": "0" * 64,
            "consent_timestamp": "2026-10-01T14:03:22Z", "adult_confirmed": True, "language": ["ar"],
            "dialect_primary": "Jordanian",
            "scopes": {"ml_training": True, "voice_anonymization_rnd": True, "commercial_product_development": True,
                       "model_evaluation": True, "derivative_model_training": True, "audio_redistribution": False},
            "withdrawal": {"status": "active", "withdrawn_at": None}, "retention_until": "2028-01-01",
            "withdrawal_token_sha256": "a" * 64, "collector": "synthetic-test"}


def session(spk, cid, sid, recorded_at="2026-10-02"):
    return {"speaker_id": spk, "consent_id": cid, "session_id": sid, "language": "ar", "dialect": "Jordanian",
            "recording_device": "phone_android", "environment": "quiet_room", "sampling_rate_hz": SR,
            "recorded_at": recorded_at, "duration_s": 10.0, "consent_scope_snapshot": "b" * 64,
            "withdrawal_status": "active"}


class _ImportFixture(unittest.TestCase):
    """Runs the importer once per test class on the synthetic incoming tree (no tests here)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        inc, store = os.path.join(cls.tmp, "in"), os.path.join(cls.tmp, "consents.jsonl")
        recs = []
        for i in range(6):
            cid, spk = f"CNS-{i:012x}", f"own_recordings:spk-{i:010x}"
            recs.append(consent(cid, spk))
            for k in range(2):
                sid = f"ses-{i:04x}{k:04x}"
                sd = os.path.join(inc, cid, sid)
                os.makedirs(sd)
                json.dump(session(spk, cid, sid), open(os.path.join(sd, "session.json"), "w"))
                name = f"voice_2026-10-0{k + 2}_14-03-2{i}"                  # dated, Telegram-style
                sf.write(os.path.join(sd, name + ".wav"), voiced(2.0, 100 + 9 * i + 3 * k, i * 7 + k), SR)
                open(os.path.join(sd, name + ".txt"), "w", encoding="utf-8").write("مرحبا كيف حالك اليوم يا صديقي")
        with open(store, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        s0 = os.path.join(inc, "CNS-000000000000", "ses-00000000")
        open(os.path.join(s0, "audio_20261002_140322.wav"), "wb").write(b"RIFF\x00\x00garbage")   # corrupt, dated
        s1 = os.path.join(inc, "CNS-000000000001", "ses-00010000")
        shutil.copy(os.path.join(s0, "voice_2026-10-02_14-03-20.wav"), os.path.join(s1, "dup_1759413802.wav"))
        shutil.copy(os.path.join(s0, "voice_2026-10-02_14-03-20.txt"), os.path.join(s1, "dup_1759413802.txt"))
        s4 = os.path.join(inc, "CNS-000000000004", "ses-00040001")                  # transcript with personal data
        sf.write(os.path.join(s4, "x.wav"), voiced(2.0, 271, 99), SR)
        open(os.path.join(s4, "x.txt"), "w", encoding="utf-8").write("اسمي سارة ورقمي ٠٧٩١٢٣٤٥٦٧ شكرا جزيلا لكم")
        os.makedirs(os.path.join(inc, "CNS-000000000002", "2026-10-02_evening"))   # unvalidated dir name
        os.makedirs(os.path.join(inc, "2026-10-02_batch"))                          # unknown consent dir
        bad = os.path.join(inc, "CNS-000000000003", "ses-0003000f")                 # invalid metadata
        os.makedirs(bad)
        json.dump(session("own_recordings:spk-0000000003", "CNS-000000000003", "ses-0003000f",
                          recorded_at="2026-10-02T14:03:22"), open(os.path.join(bad, "session.json"), "w"))
        cls.inc, cls.out, cls.store = inc, os.path.join(cls.tmp, "out"), store
        cls.source_hashes = {hashlib.sha256(open(os.path.join(dp, f), "rb").read()).hexdigest()
                             for dp, _, fs in os.walk(inc) for f in fs if f.endswith(".wav")}
        cls.rep = AI.run(inc, store, cls.out, "own-ar-c9", "research", fractions=(0.5, 0.25, 0.25))
        with open(os.path.join(cls.out, "import_report.json"), encoding="utf-8") as f:
            cls.report_text = f.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)
        shutil.rmtree(cls.out + ".provenance", ignore_errors=True)


class ImportOutputs(_ImportFixture):

    def test_report_has_no_source_names_dates_or_quoted_values(self):
        for hint in SOURCE_HINTS:
            self.assertNotIn(hint, self.report_text, hint)
        report = json.loads(self.report_text)
        hashes = report.pop("manifests")            # sha256 digests of the manifests: random hex, checked as such
        for h in hashes.values():
            self.assertRegex(h, r"^[0-9a-f]{64}$")
        masked = BM._RANDOM_ID.sub("<id>", json.dumps(report, ensure_ascii=False))
        self.assertIsNone(BM.finer_than_month(masked))
        self.assertNotIn("14:03:22", self.report_text)
        self.assertIn("'<value>'", self.report_text)                 # the invalid recorded_at was scrubbed

    def test_positional_references_point_at_the_right_entries(self):
        rej = self.rep["rejected"]
        names = sorted(f for f in os.listdir(os.path.join(self.inc, "CNS-000000000000", "ses-00000000"))
                       if f.endswith(AI.AUDIO_EXT))
        corrupt = [v for v in rej["corrupt"] if v.startswith("CNS-000000000000/ses-00000000/")]
        self.assertEqual(len(corrupt), 1)
        idx = int(re.search(r"file#(\d+)", corrupt[0]).group(1))
        self.assertEqual(names[idx], "audio_20261002_140322.wav")          # operator can locate the file
        entries = sorted(os.listdir(self.inc))
        no_consent = rej["no consent record"]
        self.assertEqual(no_consent, [f"dir#{entries.index('2026-10-02_batch')}"])
        self.assertEqual(rej["session without session.json"], ["CNS-000000000002/dir#0"])
        self.assertTrue(rej["invalid session metadata"][0].startswith("CNS-000000000003/ses-0003000f: "))

    def test_functional_results_unchanged(self):
        self.assertEqual(self.rep["accepted"], 12)                       # 6 speakers x 2 sessions
        self.assertEqual([d["type"] for d in self.rep["duplicates"]], ["exact"])
        self.assertEqual(set(self.rep["rejected"]), {"corrupt", "no consent record", "session without session.json",
                                                     "invalid session metadata", "transcript needs human review"})
        BM.verify(os.path.join(self.out, "manifests"))

    def test_lineage_ids_and_paths_intact(self):
        recs = [json.loads(l) for l in open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8")]
        rows = [json.loads(l) for s in ("train", "valid", "test")
                for l in open(os.path.join(self.out, "manifests", f"{s}.jsonl"), encoding="utf-8")]
        self.assertEqual(len(rows), 12)
        by_id = {r["recording_id"]: r for r in recs}
        for row in rows:
            self.assertTrue(os.path.exists(row["path"]))
            rid = os.path.splitext(os.path.basename(row["path"]))[0]
            self.assertRegex(rid, r"^rec-[0-9a-f]{16}$")
            self.assertEqual((by_id[rid]["speaker_id"], by_id[rid]["session_id"], by_id[rid]["consent_id"]),
                             (row["speaker"], row["session"], row["consent_id"]))
            self.assertEqual(row["path"], os.path.join(self.out, "audio", row["speaker"].split(":")[1], row["session"],
                                                       rid + ".flac"))
        rec_text = open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8").read()
        for hint in ("voice_2026", "audio_2026", "dup_17", "14-03-2", "recorded_at"):
            self.assertNotIn(hint, rec_text)



class Provenance(_ImportFixture):
    """source_sha256 (hash of the file as received) is kept only in the restricted provenance
    record outside the dataset directory."""

    def prov(self):
        return os.path.join(self.out + ".provenance", "provenance.jsonl")

    def test_source_hashes_absent_from_dataset_directory(self):
        self.assertEqual(len(self.source_hashes), 14)     # 15 source files, one an exact byte copy
        for dp, _, fs in os.walk(self.out):
            for f in fs:
                if f.endswith((".json", ".jsonl", ".md")):
                    text = open(os.path.join(dp, f), encoding="utf-8").read()
                    self.assertNotIn("source_sha256", text, f)
                    for h in self.source_hashes:
                        self.assertNotIn(h, text, f)

    def test_restricted_provenance_record(self):
        p = self.prov()
        self.assertNotEqual(os.path.commonpath([os.path.abspath(p), os.path.abspath(self.out)]),
                            os.path.abspath(self.out))
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode), 0o700)
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        recs = {json.loads(l)["recording_id"]: json.loads(l)
                for l in open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8")}
        self.assertEqual({r["recording_id"] for r in rows}, set(recs))     # one per accepted recording
        prov_schema = json.load(open(os.path.join(DATA, "recording_provenance_schema.json")))
        rec_schema = json.load(open(os.path.join(DATA, "recording_schema.json")))
        for r in rows:
            self.assertEqual(CO._validate(r, prov_schema), [])
            self.assertEqual(r["consent_id"], recs[r["recording_id"]]["consent_id"])   # deletable per participant
            self.assertIn(r["source_sha256"], self.source_hashes)
        for r in recs.values():
            self.assertEqual(CO._validate(r, rec_schema), [])

    def test_provenance_inside_dataset_refused(self):
        with self.assertRaises(ValueError):
            AI._restricted_dir(self.out, os.path.join(self.out, "prov"))


class TranscriptReview(_ImportFixture):
    def test_transcript_with_indicators_held_back_without_text(self):
        held = self.rep["rejected"]["transcript needs human review"]
        self.assertEqual(len(held), 1)
        names = sorted(f for f in os.listdir(os.path.join(self.inc, "CNS-000000000004", "ses-00040001"))
                       if f.endswith(AI.AUDIO_EXT))
        self.assertTrue(held[0].startswith(f"CNS-000000000004/ses-00040001/file#{names.index('x.wav')}: indicators "))
        self.assertIn("self_introduction", held[0])
        self.assertIn("phone_or_id_number", held[0])
        for token in ("سارة", "0791234567", "٠٧٩١٢٣٤٥٦٧", "اسمي"):
            self.assertNotIn(token, self.report_text)
        recs = open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8").read()
        self.assertNotIn("سارة", recs)
        audio = os.path.join(self.out, "audio", "spk-0000000004", "ses-00040001")
        self.assertEqual(len(os.listdir(audio)), 1)                      # only the accepted file, not x.wav


class PositionalReferences(_ImportFixture):
    """file#N / dir#N let an operator with access to the incoming store find a file; nothing in the
    dataset outputs maps them to an accepted recording."""

    def test_refs_not_linkable_to_recordings(self):
        recs = [json.loads(l) for l in open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8")]
        rec_text = json.dumps(recs, ensure_ascii=False)
        self.assertNotIn("file#", rec_text)
        self.assertNotIn("dir#", rec_text)
        for r in recs:                                        # report names no recording id or audio hash
            self.assertNotIn(r["recording_id"], self.report_text)
            self.assertNotIn(r["sha256"], self.report_text)
        for split in ("train", "valid", "test"):
            m = open(os.path.join(self.out, "manifests", f"{split}.jsonl"), encoding="utf-8").read()
            self.assertNotIn("file#", m)
            self.assertNotIn("dir#", m)


class ProvenanceHardening(_ImportFixture):
    """Probe-found defects (round 4): a symlinked provenance directory or file, or a pre-existing
    0644 file, could put source hashes inside the dataset directory or leave them world-readable."""

    def run_import(self, out, prov=None):
        return AI.run(self.inc, self.store, out, "v", "research", fractions=(0.5, 0.25, 0.25), provenance_dir=prov)

    def test_symlinked_or_nested_provenance_dir_refused(self):
        out = os.path.join(self.tmp, "o1")
        os.makedirs(os.path.join(out, "inside"))
        link = os.path.join(self.tmp, "plink")
        os.symlink(os.path.join(out, "inside"), link)
        parent_link = os.path.join(self.tmp, "parent_link")
        os.symlink(out, parent_link)
        elsewhere = os.path.join(self.tmp, "elsewhere")
        os.makedirs(elsewhere)
        outside_link = os.path.join(self.tmp, "plink_out")
        os.symlink(elsewhere, outside_link)              # even a link that points outside: links are refused
        for prov in (link, outside_link, os.path.join(parent_link, "prov"), out, os.path.join(out, "x", "..", "p"),
                     os.path.dirname(out)):
            with self.subTest(prov=prov), self.assertRaises(ValueError):
                self.run_import(out, prov)
        self.assertFalse(os.path.exists(os.path.join(out, "inside", AI.PROVENANCE_FILE)))

    def test_preexisting_permissive_or_symlinked_file_replaced_not_followed(self):
        out = os.path.join(self.tmp, "o2")
        prov = os.path.join(self.tmp, "p2")
        os.makedirs(prov)
        os.makedirs(out)
        target = os.path.join(out, "leak.jsonl")
        open(target, "w").close()
        os.symlink(target, os.path.join(prov, AI.PROVENANCE_FILE))
        os.chmod(prov, 0o755)
        self.run_import(out, prov)
        p = os.path.join(prov, AI.PROVENANCE_FILE)
        self.assertEqual(os.path.getsize(target), 0)                         # link target untouched
        self.assertFalse(os.path.islink(p))
        self.assertEqual(stat.S_IMODE(os.lstat(p).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(prov).st_mode), 0o700)
        os.chmod(p, 0o644)                                                    # permissive leftover
        self.run_import(out, prov)
        self.assertEqual(stat.S_IMODE(os.lstat(p).st_mode), 0o600)

    def test_modes_hold_under_permissive_umask(self):
        old = os.umask(0)
        try:
            out = os.path.join(self.tmp, "o3")
            self.run_import(out)
        finally:
            os.umask(old)
        p = out + ".provenance"
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(p, AI.PROVENANCE_FILE)).st_mode), 0o600)

    def test_write_failure_leaves_no_partial_or_exported_file(self):
        out = os.path.join(self.tmp, "o4")
        prov = os.path.join(self.tmp, "p4")
        self.run_import(out, prov)
        before = open(os.path.join(prov, AI.PROVENANCE_FILE), "rb").read()
        out2 = os.path.join(self.tmp, "o5")
        real_fsync = os.fsync

        def failing(fd):
            raise OSError("disk full (injected)")
        AI.os.fsync = failing
        try:
            with self.assertRaises(OSError):
                self.run_import(out2, prov)
        finally:
            AI.os.fsync = real_fsync
        self.assertEqual(open(os.path.join(prov, AI.PROVENANCE_FILE), "rb").read(), before)   # old record intact
        self.assertEqual([f for f in os.listdir(prov) if f.endswith(".tmp")], [])
        for name in ("recordings.jsonl", "import_report.json", "manifests"):
            self.assertFalse(os.path.exists(os.path.join(out2, name)), name)

    def test_provenance_dirs_are_git_ignored(self):
        root = os.path.dirname(os.path.dirname(HERE))
        r = subprocess.run(["git", "-C", root, "check-ignore", "-q", "data/own_ar_v1.provenance/provenance.jsonl"])
        self.assertEqual(r.returncode, 0)


class Purge(_ImportFixture):
    """Withdrawal after import: purge_participant removes the participant from every import output
    and from the restricted provenance record; nothing linkable stays in either directory."""

    def secrets_of(self, cid):
        recs = [json.loads(l) for l in open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8")]
        prov = [json.loads(l) for l in open(os.path.join(self.out + ".provenance", AI.PROVENANCE_FILE))]
        mine = [r for r in recs if r["consent_id"] == cid]
        vals = {cid, mine[0]["speaker_id"].split(":")[1]}
        vals |= {r["recording_id"] for r in mine} | {r["sha256"] for r in mine}
        vals |= {p["source_sha256"] for p in prov if p["consent_id"] == cid}
        return vals, len(recs), mine

    def scan(self, needles):
        hits = []
        for root in (self.out, self.out + ".provenance"):
            for dp, _, fs in os.walk(root):
                for f in fs:
                    p = os.path.join(dp, f)
                    blob = open(p, "rb").read().decode("utf-8", "ignore") + p
                    hits += [(os.path.relpath(p, self.tmp), n) for n in needles if n in blob]
        return hits

    def test_purge_removes_everything_linkable_and_is_idempotent(self):
        cid = "CNS-000000000000"                       # accepted x2, a corrupt rejection, a duplicate pair
        needles, n_before, mine = self.secrets_of(cid)
        self.assertTrue(self.scan(needles))
        counts = AI.purge_participant(self.out, cid)
        self.assertEqual(counts["recordings"], 2)
        self.assertEqual(counts["provenance_rows"], 2)
        self.assertEqual(counts["manifest_rows"], 2)
        self.assertEqual(counts["audio_dirs"], 1)
        self.assertGreaterEqual(counts["report_entries"], 2)
        self.assertEqual(self.scan(needles), [])
        BM.verify(os.path.join(self.out, "manifests"))
        rep = json.load(open(os.path.join(self.out, "import_report.json"), encoding="utf-8"))
        self.assertEqual(rep["accepted"], n_before - 2)
        self.assertNotIn("purged_participants", rep)
        recs = [json.loads(l) for l in open(os.path.join(self.out, "recordings.jsonl"), encoding="utf-8")]
        self.assertEqual(len(recs), n_before - 2)
        self.assertEqual(AI.purge_participant(self.out, cid),
                         {"recordings": 0, "audio_dirs": 0, "manifest_rows": 0, "report_entries": 0, "provenance_rows": 0,
                          "stale_temps": 0})

    def test_purge_resumes_after_a_crash(self):
        cid = "CNS-000000000001"
        needles, _, _ = self.secrets_of(cid)
        real = AI._write_restricted

        def crash(*a, **k):
            raise OSError("power loss (injected)")
        AI._write_restricted = crash
        try:
            with self.assertRaises(OSError):
                AI.purge_participant(self.out, cid)
        finally:
            AI._write_restricted = real
        self.assertTrue(self.scan(needles))                 # partially done, recordings.jsonl still lists it
        AI.purge_participant(self.out, cid)
        self.assertEqual(self.scan(needles), [])
        BM.verify(os.path.join(self.out, "manifests"))

    def test_purge_when_recordings_already_lost_the_participant(self):
        """recordings.jsonl no longer lists the participant (older partial purge, manual edit):
        the speaker is still found through the manifests and the audio is still removed."""
        cid = "CNS-000000000002"
        needles, _, _ = self.secrets_of(cid)
        p = os.path.join(self.out, "recordings.jsonl")
        keep = [l for l in open(p, encoding="utf-8") if json.loads(l)["consent_id"] != cid]
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(keep)
        counts = AI.purge_participant(self.out, cid)
        self.assertEqual(counts["audio_dirs"], 1)
        self.assertEqual(self.scan(needles), [])

    def test_bad_consent_id_refused(self):
        for cid in ("../x", "CNS-0", "", "spk-0000000000"):
            with self.subTest(cid=cid), self.assertRaises(ValueError):
                AI.purge_participant(self.out, cid)


class CliOutputs(_ImportFixture):
    def test_cli_prints_no_text_names_or_hashes(self):
        script = os.path.join(os.path.dirname(HERE), "datasets", "arabic_import.py")
        r = subprocess.run([sys.executable, script, "--incoming", self.inc, "--consents", self.store,
                            "--out", os.path.join(self.tmp, "cli"), "--dataset-version", "v", "--licence-path", "research"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        both = r.stdout + r.stderr
        for token in ("سارة", "0791234567", "٠٧٩١٢٣٤٥٦٧", "voice_", "audio_2026", "dup_17") + tuple(self.source_hashes):
            self.assertNotIn(token, both)


def _purge_in_child(out, cid, point, nth):
    """Run purge_participant in a forked child that dies with os._exit (no cleanup, no finally)
    at the nth call of `point`: "replace" = just before an atomic replace (manifest files,
    report, provenance, recordings), "rmtree" = after removing one audio file. -> exit code
    (0 = died at that point, 2 = finished before reaching it)."""
    pid = os.fork()
    if pid == 0:
        code = 3
        try:
            calls = {"n": 0}
            if point == "replace":
                real = os.replace

                def replace(*a, **k):
                    calls["n"] += 1
                    if calls["n"] == nth:
                        os._exit(0)
                    return real(*a, **k)
                os.replace = replace
            else:
                def rmtree(path, *a, **k):
                    for dp, _, fs in os.walk(path):
                        for f in fs:
                            os.unlink(os.path.join(dp, f))
                            os._exit(0)
                shutil.rmtree = rmtree
            AI.purge_participant(out, cid)
            code = 2
        except BaseException:
            code = 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


class PurgeHardCrash(_ImportFixture):
    """Question 1 of the closing review: a purge interrupted by a HARD crash at any step (fork +
    os._exit: no finally, no cleanup) is completed by a rerun; afterwards nothing linkable to the
    participant remains in the dataset or provenance directory, and no other participant's data
    is lost."""
    CID = "CNS-000000000000"

    def scenario(self, tag):
        out = os.path.join(self.tmp, f"crash-{tag}")
        shutil.copytree(self.out, out, symlinks=True)
        shutil.copytree(self.out + ".provenance", out + ".provenance", symlinks=True)
        return out

    def needles(self, out):
        recs = [json.loads(l) for l in open(os.path.join(out, "recordings.jsonl"), encoding="utf-8")]
        prov = [json.loads(l) for l in open(os.path.join(out + ".provenance", AI.PROVENANCE_FILE))]
        mine = [r for r in recs if r["consent_id"] == self.CID]
        v = {self.CID, mine[0]["speaker_id"].split(":")[1]} | {r["recording_id"] for r in mine}
        v |= {r["sha256"] for r in mine} | {p["source_sha256"] for p in prov if p["consent_id"] == self.CID}
        return v, len(recs), len(prov)

    def leftovers(self, out, needles):
        hits = []
        for root in (out, out + ".provenance"):
            for dp, dn, fs in os.walk(root):
                for f in fs + dn:
                    p = os.path.join(dp, f)
                    blob = p + (open(p, "rb").read().decode("utf-8", "ignore") if os.path.isfile(p) else "")
                    hits += [(os.path.relpath(p, out), n) for n in needles if n in blob]
        return hits

    def check_final(self, out, needles, n_recs, n_prov):
        self.assertEqual(self.leftovers(out, needles), [])
        BM.verify(os.path.join(out, "manifests"))
        self.assertEqual(sum(1 for _ in open(os.path.join(out, "recordings.jsonl"))), n_recs - 2)
        self.assertEqual(sum(1 for _ in open(os.path.join(out + ".provenance", AI.PROVENANCE_FILE))), n_prov - 2)
        rows = sum(1 for s in ("train", "valid", "test", "attacker_train")
                   for _ in open(os.path.join(out, "manifests", f"{s}.jsonl")))
        self.assertEqual(rows, n_recs - 2)                         # no other participant's row lost
        rep = json.load(open(os.path.join(out, "import_report.json"), encoding="utf-8"))
        self.assertEqual(rep["accepted"], n_recs - 2)

    def test_crash_before_every_atomic_replace_then_rerun(self):
        n = 0
        while True:
            n += 1
            out = self.scenario(f"r{n}")
            needles, n_recs, n_prov = self.needles(out)
            code = _purge_in_child(out, self.CID, "replace", n)
            if code == 2:                                           # passed every replace point
                break
            self.assertEqual(code, 0, n)
            AI.purge_participant(out, self.CID)                     # rerun completes the purge
            self.check_final(out, needles, n_recs, n_prov)
        self.assertGreaterEqual(n, 10)                              # 7 manifest files + report + provenance + recordings

    def test_crash_inside_audio_deletion_then_rerun(self):
        out = self.scenario("rm")
        needles, n_recs, n_prov = self.needles(out)
        self.assertEqual(_purge_in_child(out, self.CID, "rmtree", 1), 0)
        AI.purge_participant(out, self.CID)
        self.check_final(out, needles, n_recs, n_prov)

    def test_hash_mismatch_without_marker_is_not_blessed(self):
        """A manifest edited by someone else (no purge marker) is not rewritten by a purge."""
        out = self.scenario("edit")
        p = os.path.join(out, "manifests", "train.jsonl")
        with open(p, "a", encoding="utf-8") as f:
            f.write(open(os.path.join(out, "manifests", "test.jsonl")).readline())
        before = {f: open(os.path.join(out, "manifests", f), "rb").read() for f in os.listdir(os.path.join(out, "manifests"))}
        with self.assertRaises(ValueError):
            AI.purge_participant(out, "CNS-000000000005")
        after = {f: open(os.path.join(out, "manifests", f), "rb").read() for f in os.listdir(os.path.join(out, "manifests"))}
        self.assertEqual(before, after)                        # nothing rewritten, the edit stays visible
        self.assertTrue(os.path.isdir(os.path.join(out, "audio", "spk-0000000005")))   # nothing deleted either
        with self.assertRaises(BM.LeakageError):
            BM.verify(os.path.join(out, "manifests"))

    def test_stale_temps_from_a_crashed_import_are_purged(self):
        out = self.scenario("stale")
        needles, n_recs, n_prov = self.needles(out)
        pdir = out + ".provenance"
        shutil.copy(os.path.join(pdir, AI.PROVENANCE_FILE), os.path.join(pdir, f".{AI.PROVENANCE_FILE}.deadbeef.tmp"))
        shutil.copy(os.path.join(out, "recordings.jsonl"), os.path.join(out, "recordings.jsonl.deadbeef.tmp"))
        shutil.copytree(os.path.join(out, "manifests"), os.path.join(out, "manifests.deadbeef.tmp"))
        counts = AI.purge_participant(out, self.CID)
        self.assertEqual(counts["stale_temps"], 3)
        self.check_final(out, needles, n_recs, n_prov)


class HeldTranscriptPaths(_ImportFixture):
    """Question 3 of the closing review: a recording held back for its transcript reaches no
    training input through the paths that exist. Training reads only manifests, and only after
    build_manifests.verify() (trainers/requirements.py, datasets/mix.py, the orchestrator
    preflight)."""
    HELD = ("CNS-000000000004", "ses-00040001", "x.wav")

    def held_ids(self):
        y, _, _ = AI.check_audio(os.path.join(self.inc, *self.HELD))
        sha = AI.pcm_sha(y)
        return {sha, "rec-" + sha[:16]}

    def test_held_recording_absent_from_every_output(self):
        ids = self.held_ids()
        for root in (self.out, self.out + ".provenance"):
            for dp, dn, fs in os.walk(root):
                for f in fs + dn:
                    p = os.path.join(dp, f)
                    blob = p + (open(p, "rb").read().decode("utf-8", "ignore") if os.path.isfile(p) else "")
                    for i in ids:
                        self.assertNotIn(i, blob, p)

    def test_generic_builder_cannot_take_the_incoming_store(self):
        import soundfile
        with self.assertRaises(soundfile.LibsndfileError):          # as delivered: the corrupt file stops it
            BM.build_generic(self.inc, "own_recordings", "ar")
        inc = os.path.join(self.tmp, "inc-clean")
        shutil.copytree(self.inc, inc)
        os.remove(os.path.join(inc, "CNS-000000000000", "ses-00000000", "audio_20261002_140322.wav"))
        rows = BM.build_generic(inc, "own_recordings", "ar")         # incoming layout: <consent>/<session>/<file>
        self.assertTrue(any(r["path"].endswith("x.wav") for r in rows))
        split, _ = BM.assign_splits(rows, 5, fractions=(0.5, 0.25, 0.25))
        with self.assertRaises(BM.LeakageError) as cm:                # speaker ids are not pseudonyms, text flagged
            BM.assert_no_leakage(split)
        self.assertIn("participant speaker is not a random schema id", str(cm.exception))

    def test_generic_builder_over_import_audio_has_no_held_recording(self):
        rows = BM.build_generic(os.path.join(self.out, "audio"), "own_recordings", "ar")
        names = {os.path.splitext(os.path.basename(r["path"]))[0] for r in rows}
        self.assertFalse(names & self.held_ids())
        self.assertEqual(len(rows), self.rep["accepted"])

    def reimport(self, tag, transcript):
        inc = os.path.join(self.tmp, f"inc-{tag}")
        shutil.copytree(self.inc, inc)
        tp = os.path.join(inc, self.HELD[0], self.HELD[1], "x.txt")
        if transcript is None:
            os.remove(tp)
        else:
            open(tp, "w", encoding="utf-8").write(transcript)
        out = os.path.join(self.tmp, f"out-{tag}")
        return AI.run(inc, self.store, out, "v", "research", fractions=(0.5, 0.25, 0.25)), out

    def test_reimport_without_transcript_still_rejected(self):
        rep, _ = self.reimport("notext", None)
        self.assertEqual(rep["accepted"], self.rep["accepted"])
        self.assertTrue(any(v.startswith("CNS-000000000004/ses-00040001/") for v in rep["rejected"]["transcript"]))

    def test_corrected_transcript_readmits_the_audio(self):
        """STATED LIMIT, pinned by this test: the screen reads transcripts, not audio. Replacing
        the transcript readmits a recording whose AUDIO may still contain the spoken personal
        data, so a correction is only acceptable after a human has reviewed the audio too."""
        rep, out = self.reimport("fixed", "شكرا جزيلا لكم على المشاركة اليوم")
        self.assertEqual(rep["accepted"], self.rep["accepted"] + 1)
        recs = open(os.path.join(out, "recordings.jsonl"), encoding="utf-8").read()
        self.assertTrue(any(i in recs for i in self.held_ids()))


if __name__ == "__main__":
    unittest.main(verbosity=1)
