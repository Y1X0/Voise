"""Tests for world_clarity_kernel.py (Kaggle wrapper of the WORLD clarity screen). Mocks and synthetic
fixtures only: no network, no Kaggle, no dataset, no ASR model; the real screen main() is never called.
"""
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world_clarity_kernel as K  # noqa: E402
import world_clarity_screen as W  # noqa: E402

REPO = K.ROOT
FROZEN_PREREG = ("# WORLD clarity screen — pre-registration (v1.0, synthetic test fixture)\n\n"
                 "**Status: frozen by the owner.**\n\n## 1. Purpose\n\ntext\n\n## Freeze record\n\n"
                 "| Field | Value |\n|---|---|\n| Approved by | test fixture |\n")


def repo_bytes(root, path):
    with open(os.path.join(REPO, path), "rb") as f:
        return f.read()


def load_module(path, name="built_run"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                           # __name__ != "__main__": nothing runs
    return m


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "root")
        for p in K.EMBED:
            os.makedirs(os.path.dirname(os.path.join(self.root, p)), exist_ok=True)
            shutil.copyfile(os.path.join(REPO, p), os.path.join(self.root, p))
        with open(os.path.join(self.root, K.PREREG), "w", encoding="utf-8") as f:
            f.write(FROZEN_PREREG)
        self.prereg_sha = K.sha256(FROZEN_PREREG.encode())
        self.out = os.path.join(self.tmp, "kernel")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, **kw):
        a = dict(root=self.root, out=self.out, kernel_id="y1x00r/voise-world-clarity-screen",
                 approval="owner approved the run (test)", prereg_sha256=self.prereg_sha, committed=repo_bytes)
        a.update(kw)
        return K.build(**a)

    def payload(self):
        if not hasattr(self, "_payload"):
            self.build()
            self._payload = load_module(os.path.join(self.out, "run.py")).PAYLOAD
        return json.loads(json.dumps(self._payload))


# ------------------------------------------------------------------ build / arguments
class TestBuild(Fixture):
    def test_cli_argument_handling(self):
        for argv in ([], ["push"], ["build"], ["build", "--out", "x", "--kernel-id", "a/bcdef", "--prereg-sha256", "0"]):
            with self.assertRaises(SystemExit) as c, mock.patch("sys.stderr"):
                K.cli(argv)
            self.assertEqual(c.exception.code, 2)

    def test_real_repo_frozen_prereg_builds_only_at_its_sha(self):
        out = os.path.join(self.tmp, "real")
        sha = K.sha256(open(os.path.join(REPO, K.PREREG), "rb").read())
        with self.assertRaises(K.BuildRefused) as c:
            K.build(REPO, out, "y1x00r/voise-world-clarity-screen", "x", "0" * 64)
        self.assertEqual(len(c.exception.problems), 1)
        self.assertIn("--prereg-sha256", c.exception.problems[0])
        self.assertFalse(os.path.exists(out))
        with mock.patch("sys.stderr"):
            self.assertEqual(K.cli(["build", "--out", out, "--kernel-id", "y1x00r/voise-world-clarity-screen",
                                    "--approval", "x", "--prereg-sha256", "0" * 64]), 2)
        self.assertFalse(os.path.exists(out))
        r = K.build(REPO, out, "y1x00r/voise-world-clarity-screen", "x", sha)   # real git check vs VOISE_COMMIT
        self.assertEqual(r["files"][K.PREREG], sha)
        p = load_module(os.path.join(out, "run.py")).PAYLOAD
        self.assertEqual(K.payload_problems(p), [])

    def test_build_ok(self):
        r = self.build()
        self.assertEqual(sorted(os.listdir(self.out)), ["kernel-metadata.json", "run.py"])
        meta = json.load(open(os.path.join(self.out, "kernel-metadata.json")))
        self.assertEqual((meta["id"], meta["code_file"], meta["kernel_type"], meta["is_private"], meta["enable_gpu"],
                          meta["enable_internet"], meta["dataset_sources"]),
                         ("y1x00r/voise-world-clarity-screen", "run.py", "script", True, True, True, []))
        self.assertTrue(r["push_command"].startswith("kaggle kernels push -p "))
        src = open(os.path.join(self.out, "run.py"), encoding="utf-8").read()
        self.assertNotIn("\nPAYLOAD = None  # set by build()\n", src)
        self.assertIn("\nPAYLOAD = json.loads(base64.b64decode(", src)
        p = load_module(os.path.join(self.out, "run.py")).PAYLOAD
        self.assertEqual(K.payload_problems(p), [])
        self.assertEqual(set(p["files"]), set(K.EMBED))
        for path, v in p["files"].items():
            self.assertEqual(base64.b64decode(v["b64"]), open(os.path.join(self.root, path), "rb").read())
        self.assertEqual(p["approval"], "owner approved the run (test)")
        self.assertEqual(p["voise_commit"], K.VOISE_COMMIT)

    def test_refusals(self):
        cases = [
            (dict(kernel_id="Y1X0/Voise Screen"), "kernel id"),
            (dict(kernel_id="noslash"), "kernel id"),
            (dict(approval="   "), "approval"),
            (dict(prereg_sha256="0" * 64), "--prereg-sha256"),
            (dict(committed=lambda root, path: b"other"), "differs from Voise commit"),
            (dict(committed=mock.Mock(side_effect=subprocess.CalledProcessError(128, "git"))), "not readable"),
        ]
        for kw, needle in cases:
            with self.assertRaises(K.BuildRefused) as c:
                self.build(**kw)
            self.assertTrue(any(needle in p for p in c.exception.problems), (kw, c.exception.problems))
            self.assertFalse(os.path.exists(self.out))

    def test_refuses_nonempty_out_changed_criteria_missing_file(self):
        os.makedirs(self.out)
        open(os.path.join(self.out, "keep"), "w").close()
        with self.assertRaises(K.BuildRefused):
            self.build()
        self.assertEqual(os.listdir(self.out), ["keep"])
        shutil.rmtree(self.out)
        crit = os.path.join(self.root, K.CRITERIA)
        with open(crit, "ab") as f:
            f.write(b" ")
        with self.assertRaises(K.BuildRefused) as c:
            self.build(committed=lambda root, path: open(os.path.join(root, path), "rb").read())
        self.assertTrue(any("frozen" in p for p in c.exception.problems))
        os.remove(os.path.join(self.root, K.SCREEN))
        with self.assertRaises(K.BuildRefused) as c:
            self.build()
        self.assertTrue(any(p.startswith("missing") for p in c.exception.problems))

    def test_draft_detection(self):
        self.assertEqual(K.prereg_frozen_problems(FROZEN_PREREG), [])
        real = open(os.path.join(REPO, K.PREREG), encoding="utf-8").read()
        self.assertEqual(K.prereg_frozen_problems(real), [])                    # frozen v1.1
        draft = real.replace("(v1.1, frozen 2026-10-18)", "(DRAFT v0.4, 2026-10-18)").replace(
            "| Approval date | 2026-10-18 |", "| Approval date | (pending) |")
        self.assertEqual(len(K.prereg_frozen_problems(draft)), 2)
        self.assertEqual(K.prereg_frozen_problems("# x\n\nStatus: frozen\n"), ["pre-registration has no freeze record"])

    def test_committed_inputs_match_git(self):
        for p, kind in K.EMBED.items():
            if kind == "commit":
                self.assertEqual(K.git_committed_bytes(REPO, p), repo_bytes(REPO, p), p)

    def test_script_without_payload_refuses(self):
        r = subprocess.run([sys.executable, os.path.join(HERE, "world_clarity_kernel.py")], capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no embedded payload", r.stderr)


# ------------------------------------------------------------------ kernel runtime
class FakeScreen:
    def __init__(self, tmp, main):
        self.WORK, self.HF_CACHE = os.path.join(tmp, "voise_screen"), os.path.join(tmp, "voise_screen_hf")
        self.main = main
        self.cleanup = W.cleanup


class TestKernel(Fixture):
    def setUp(self):
        super().setUp()
        self.code_root = os.path.join(self.tmp, "code")
        self.outdir = os.path.join(self.tmp, "working")
        self.logs = []
        self.events = []
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("HF_HOME", "VOISE_SCREEN_APPROVED"):
            os.environ.pop(k, None)
        mods = mock.patch.dict(sys.modules)
        mods.start()
        self.addCleanup(mods.stop)
        for m in K.HF_MODULES:
            sys.modules.pop(m, None)

    def run_kernel(self, main, payload=None, install=None, load_screen=None):
        self.screen = FakeScreen(self.tmp, main)
        return K.kernel_main(payload if payload is not None else self.payload(), code_root=self.code_root,
                             outdir=self.outdir, install=install or (lambda: self.events.append("install")),
                             load_screen=load_screen or (lambda root: self.events.append("load") or self.screen),
                             log=self.logs.append)

    def results(self):
        return json.load(open(os.path.join(self.outdir, "results.json")))

    def write_results(self, extra=()):
        def main():
            self.events.append(("main", os.environ.get("HF_HOME"), os.environ.get("VOISE_SCREEN_APPROVED")))
            os.makedirs(os.path.join(self.screen.WORK, "src"))
            os.makedirs(self.screen.HF_CACHE)
            with open(os.path.join(self.outdir, "results.json"), "w") as f:
                json.dump(dict(status="DONE", result="PASS"), f)
            for name in extra:
                os.makedirs(os.path.join(self.outdir, name)) if name.endswith("/") else \
                    open(os.path.join(self.outdir, name), "w").close()
        return main

    def assert_cleaned(self):
        for p in (self.screen.WORK, self.screen.HF_CACHE, self.code_root):
            self.assertFalse(os.path.exists(p), p)

    def test_success_environment_and_order(self):
        self.assertEqual(self.run_kernel(self.write_results()), 0)
        self.assertEqual(self.events, ["load", "install", ("main", self.screen.HF_CACHE, "1")])
        self.assertEqual(os.listdir(self.outdir), ["results.json"])
        self.assertEqual(self.results()["result"], "PASS")
        self.assert_cleaned()
        self.assertTrue(any(l.startswith("approval: owner approved") for l in self.logs))
        self.assertTrue(any(l.startswith("results.json sha256 ") for l in self.logs))

    def test_gate_refusals(self):
        good = self.payload()
        bad_prereg = json.loads(json.dumps(good))
        draft = b"# x (DRAFT v0.4)\n\n## Freeze record\n| Approved by | (pending) |\n"
        bad_prereg["files"][K.PREREG] = dict(sha256=K.sha256(draft), b64=base64.b64encode(draft).decode())
        bad_prereg["prereg_sha256"] = K.sha256(draft)
        tampered = json.loads(json.dumps(good))
        tampered["files"][K.SCREEN]["b64"] = base64.b64encode(b"print('other code')").decode()
        missing = json.loads(json.dumps(good))
        del missing["files"]["scripts/neural_anonymization_eval.py"]
        wrong_sha = dict(good, prereg_sha256="0" * 64)
        cases = [(dict(good, approval=" "), "approval"), (bad_prereg, "DRAFT"), (tampered, "sha256 mismatch"),
                 (missing, "file set"), (wrong_sha, "approved prereg_sha256"), (None, "no embedded payload")]
        for payload, needle in cases:
            self.logs.clear()
            self.events.clear()
            code = K.kernel_main(payload, code_root=self.code_root, outdir=self.outdir,
                                 install=lambda: self.events.append("install"),
                                 load_screen=lambda root: self.events.append("load"), log=self.logs.append)
            self.assertEqual(code, 2, needle)
            self.assertIn(needle, self.logs[0])
            self.assertEqual(self.events, [])
            self.assertFalse(os.path.exists(self.outdir) or os.path.exists(self.code_root))
            self.assertNotIn("VOISE_SCREEN_APPROVED", os.environ)
            self.assertNotIn("HF_HOME", os.environ)

    def test_refuses_if_hf_already_imported(self):
        with mock.patch.dict(sys.modules, {"transformers": object()}):
            self.assertEqual(self.run_kernel(self.write_results()), 2)
        self.assertEqual(self.events, [])
        self.assertIn("before HF_HOME", self.logs[0])

    def test_output_restriction(self):
        os.makedirs(self.outdir)
        open(os.path.join(self.outdir, "preexisting.txt"), "w").close()
        self.assertEqual(self.run_kernel(self.write_results(extra=("leak.wav", "cache/"))), 3)
        self.assertEqual(sorted(os.listdir(self.outdir)), ["preexisting.txt", "results.json"])
        self.assertTrue(any("OUTPUT_VIOLATION" in l and "leak.wav" in l and "cache" in l for l in self.logs))
        self.assert_cleaned()

    def test_install_failure(self):
        main = mock.Mock()

        def install():
            raise subprocess.CalledProcessError(1, "pip")
        self.assertEqual(self.run_kernel(main, install=install), 1)
        main.assert_not_called()
        r = self.results()
        self.assertEqual((r["status"], r["stage"], r["error_type"]), ("FAILED", "install", "CalledProcessError"))
        self.assertEqual(set(r["code_sha256"]), set(K.EMBED))
        self.assert_cleaned()

    def test_screen_exit_text_is_kept(self):
        def main():
            os.makedirs(self.screen.WORK)
            sys.exit("integrity check failed (§3): arz_Arab: rows {'dev': 1} != {'dev': 2}")
        self.assertEqual(self.run_kernel(main), 1)
        r = self.results()
        self.assertEqual((r["stage"], r["error_type"]), ("screen", "SystemExit"))
        self.assertTrue(r["exit"].startswith("integrity check failed"))
        self.assert_cleaned()

    def test_exception_message_never_recorded(self):
        def main():
            raise ValueError("كتاب بيت reference text")
        self.assertEqual(self.run_kernel(main), 1)
        r = self.results()
        self.assertEqual(r["error_type"], "ValueError")
        self.assertTrue(r["frames"])
        blob = json.dumps(r, ensure_ascii=False) + "\n".join(self.logs)
        self.assertNotIn("كتاب", blob)
        self.assertNotIn("reference text", blob)
        self.assert_cleaned()

    def test_screen_without_results_and_results_kept_on_late_failure(self):
        self.assertEqual(self.run_kernel(lambda: None), 1)
        self.assertEqual(self.results()["error_type"], "RuntimeError")
        os.remove(os.path.join(self.outdir, "results.json"))
        shutil.rmtree(self.code_root, ignore_errors=True)
        inner = self.write_results()

        def main():
            inner()
            raise OSError("late")
        self.assertEqual(self.run_kernel(main), 1)
        self.assertEqual(self.results(), dict(status="DONE", result="PASS"))   # the screen's file is not overwritten

    def test_screen_exit_zero_is_success(self):
        inner = self.write_results()

        def main():
            inner()
            sys.exit(0)
        self.assertEqual(self.run_kernel(main), 0)

    def test_setup_failure(self):
        def load(root):
            raise ImportError("broken")
        self.assertEqual(self.run_kernel(None, load_screen=load), 1)
        r = self.results()
        self.assertEqual((r["stage"], r["error_type"]), ("setup", "ImportError"))
        self.assertFalse(os.path.exists(self.code_root))
        self.assertNotIn("VOISE_SCREEN_APPROVED", os.environ)

    def test_materialised_code_loads_from_code_root(self):
        p = self.payload()
        K.materialise(p, self.code_root)
        mod = K.import_screen(self.code_root)
        self.assertTrue(mod.__file__.startswith(self.code_root))
        self.assertTrue(mod.V1_PATH.startswith(self.code_root))
        self.assertTrue(mod.WORLD_PATH.startswith(self.code_root))
        self.assertEqual(mod.norm.__code__.co_filename, mod.V1_PATH)
        self.assertEqual(mod.load_world_class().__name__, "PseudoSpeakerWorld")
        self.assertEqual((mod.WORK, mod.HF_CACHE, mod.OUTDIR), ("/tmp/voise_screen", "/tmp/voise_screen_hf",
                                                                 "/kaggle/working"))
        crit = os.path.join(mod.ROOT, "data", "acceptance_criteria.json")
        self.assertEqual(K.sha256(open(crit, "rb").read()), K.CRITERIA_SHA256)
        self.assertTrue(os.path.isfile(os.path.join(mod.HERE, "census_results.json")))


if __name__ == "__main__":
    unittest.main()
