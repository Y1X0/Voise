"""Tripwire tests for teacher gate #2 (TEACHER_GATE_OPENVOICE.md v1.2). Stdlib only; runs nothing.

They fail if the frozen protocol, the gate script, gate #1's script, the acceptance criteria or the
frozen manifest change, if the OpenVoice gate stops sharing gate #1's evaluation and gate code byte-for-byte,
or if the weight pins drift from the recorded pin run. Updating a pinned hash below is a protocol change and
needs a new pre-registered version, never an edit after results exist.
"""
import ast, hashlib, json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GATE, GATE1 = os.path.join(HERE, "openvoice_gate.py"), os.path.join(HERE, "teacher_test.py")

FROZEN_SHA256 = {
    "experiments/arabic_eval_v2/TEACHER_GATE_OPENVOICE.md": "5d2193a174b015d9513ecd39a4e9df1a86968a222724d6da3b2506d170a23888",
    "experiments/arabic_eval_v2/openvoice_gate.py": "7e8c22d7ff73a79387ec1d366df4ef88d6f7f7ac8371938f6a8920fbfd350919",
    "experiments/arabic_eval_v2/teacher_test.py": "f1d256c624a14328a39a075855f74cf4ee6ae617bfbf81e2e997d07c3885977f",
    "experiments/arabic_eval_v2/TEACHER_GATE.md": "e8ce66410450869f45e981da44b9e4c95e1a2e79016383983a7618c6d5947c18",
    "experiments/arabic_eval_v2/openvoice_pin_results.json": "ec06fd80fa9dfa5d2d34ec649c52bccb92f2298dfeaffe5b417880bb709e77bd",
    "data/acceptance_criteria.json": "58a893c21543d2075b4d962014e01cac093e9cf625af2a5b1b8a2fdd13af1aaf",
    "experiments/llvc_vs_dsp_48spk/manifest.json": "9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685",
}


def read(p):
    return open(p, encoding="utf-8").read()


def between(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def consts(src):
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            tup = [e.id for t in node.targets if isinstance(t, ast.Tuple) for e in t.elts]
            try:
                val = ast.literal_eval(node.value)
            except ValueError:
                if isinstance(node.value, ast.JoinedStr):
                    val = ast.unparse(node.value)
                else:
                    continue
            for nm in names:
                out[nm] = val
            if tup:
                out.update(zip(tup, val))
    return out


class FrozenFiles(unittest.TestCase):
    def test_hashes(self):
        for rel, sha in FROZEN_SHA256.items():
            with self.subTest(rel):
                got = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
                self.assertEqual(got, sha, f"{rel} changed: frozen material must not be edited")


class SharedWithGate1(unittest.TestCase):
    """Everything except the teacher (section 2) and the secondary section 4b must equal gate #1."""

    def setUp(self):
        self.g, self.g1 = read(GATE), read(GATE1)

    def test_helpers_and_data_checks(self):
        for start, end in (("def sh(cmd):", "raw = get(MANIFEST)"),
                           ("raw = get(MANIFEST)", 'sh("pip install'),
                           ("import numpy as np, torch, soundfile as sf", "# ---------- 2.")):
            with self.subTest(start):
                self.assertEqual(between(self.g, start, end), between(self.g1, start, end))

    def test_asr_asv_bootstrap_identical(self):
        s = between(self.g1, "# ---------- 3. intelligibility", "# ---------- 5. gates")
        self.assertIn(s, self.g)

    def test_gate_code_identical(self):
        s = self.g1[self.g1.index("# ---------- 5. gates"):]
        self.assertTrue(self.g.endswith(s))

    def test_shared_constants(self):
        a, b = consts(self.g), consts(self.g1)
        for k in ("RAW", "MANIFEST", "MANIFEST_SHA256", "V1", "AUDIT", "TRUTH", "PSN_SRC", "TARGET_SEED", "B", "SEED1", "SEED2"):
            with self.subTest(k):
                self.assertEqual(a[k], b[k])


class GatesMatchLockedCriteria(unittest.TestCase):
    def test_thresholds(self):
        crit = json.load(open(os.path.join(ROOT, "data/acceptance_criteria.json")))["criteria"]
        gates = read(GATE)[read(GATE).index("# ---------- 5. gates"):]
        t1 = crit["spontaneous_wer_rel"]["threshold"]; t3 = crit["strong_pretrained_attacker_eer"]["threshold"]
        self.assertEqual((t1, t3), (0.2, 0.35))
        self.assertIn('["rel_increase"] <= 0.20 for a in ("whisper", "mms"))', gates)       # T1, both ASRs
        self.assertIn('d["point"] >= 0.05 and d["ci95"][0] > 0 for d in d_lz.values())', gates)  # T2 (G1), all ASVs
        self.assertIn('for a in ("ecapa", "wavlm")}', gates)                               # T2 over both ASVs
        self.assertIn('["eer"] >= 0.35', gates)                                            # T3
        self.assertIn('for a in ("ecapa", "wavlm"))', gates)                               # T3 over both ASVs
        self.assertIn('T2 = all(', gates); self.assertIn('T3 = all(', gates); self.assertIn('T1 = all(', gates)


class OpenVoicePins(unittest.TestCase):
    def setUp(self):
        self.c = consts(read(GATE))
        self.pin = json.load(open(os.path.join(HERE, "openvoice_pin_results.json")))
        self.doc = read(os.path.join(HERE, "TEACHER_GATE_OPENVOICE.md"))

    def test_against_pin_run(self):
        self.assertEqual(self.c["OV_COMMIT"], self.pin["code_head"])
        self.assertEqual(self.c["HF_REV"], self.pin["hf"]["revision"])
        self.assertEqual(self.c["HF_REPO"], self.pin["hf"]["repo"])
        for p, (size, sha) in self.c["OV_FILES"].items():
            with self.subTest(p):
                self.assertEqual(self.pin["downloaded"][p], {"bytes": size, "sha256": sha})
                self.assertIn(sha, self.doc)
        self.assertTrue(self.pin["hash_match_lfs"])
        self.assertEqual(self.pin["ckpt"]["missing"], []); self.assertEqual(self.pin["ckpt"]["unexpected"], [])

    def test_conversion_settings(self):
        self.assertEqual((self.c["TAU"], self.c["OV_SEED0"], self.c["REPRO_IDX"]), (0.3, 5100, (4, 3, 2, 1, 0)))
        for m in ("openvoice.api", "openvoice.text", "openvoice.se_extractor", "unidecode", "whisper_timestamped", "wavmark"):
            self.assertIn(m, self.c["FORBIDDEN"])

    def test_script_hygiene(self):
        g = read(GATE)
        self.assertNotIn("from openvoice.api", g); self.assertNotIn("import openvoice.api", g)
        self.assertNotIn("se_extractor import", g)
        self.assertNotIn("/kaggle/working", g)
        self.assertNotIn("enable_watermark", g)
        for s in ("torch.use_deterministic_algorithms(True)", "torch.set_num_threads(1)",
                  "torch.manual_seed(OV_SEED0 + i)", 'assert max(rep.values()) == 0.0',
                  "assert not missing and not unexpected", "assert head == OV_COMMIT",
                  "assert not loaded_forbidden", 'res_type="soxr_hq"'):
            with self.subTest(s):
                self.assertIn(s, g)


class OutputsAndStops(unittest.TestCase):
    def setUp(self):
        self.g = read(GATE)

    def test_required_outputs_emitted(self):
        for s in ('R["intelligibility"] = intel', 'rel_increase_ci95=', 'teacher_minus_dsp_strong_rel', '"truth"',
                  'R["privacy"] = priv', 'eer_ci95=', 'two_level_ci95=', 'same_utterance_cos_orig_teacher_mean',
                  'top1=dict(ignorant=', 'pool_attribution=dict(', 'orig_rerun_vs_committed_max_abs',
                  'R["speed_cpu_1thread"]', 'R["weights_sha256"]', 'R["openvoice"]["repro_max_abs"]',
                  'R["per_speaker_secondary"] = per', 'R["gates"] = dict(', 'print(json.dumps(R, ensure_ascii=True))'):
            with self.subTest(s):
                self.assertIn(s, self.g)

    def test_blocked_conditions_halt_before_any_result(self):
        first_result = self.g.index("# ---------- 3. intelligibility")
        for s in ("assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256", 'assert not (test_spk & pool_spk)',
                  'assert os.path.exists(f"{EV}/', 'assert os.path.exists(f"{AU}/', "assert head == OV_COMMIT",
                  "assert len(blob) == size and got == sha", "assert not missing and not unexpected",
                  "assert max(rep.values()) == 0.0", "assert not loaded_forbidden"):
            with self.subTest(s):
                self.assertLess(self.g.index(s), first_result)
        self.assertEqual(self.g.count("RESULT_JSON_BEGIN"), 1)
        self.assertGreater(self.g.index("RESULT_JSON_BEGIN"), self.g.index('R["gates"] = dict('))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=1).result.wasSuccessful() else 1)
