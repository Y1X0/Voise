#!/usr/bin/env python3
"""Android runtime privacy tests (docs/ANDROID_RUNTIME_PRIVACY_AUDIT.md):
  * the static auditor (scripts/audit_native_runtime.py --policy strict) FAILS on a library
    with network imports, a telemetry marker, a hard-coded URL, Java network/telemetry
    classes, or a manifest network permission, and passes a clean library;
  * the pinned runtime (training/android/runtime_lock.json) contains no rejected runtime,
    and, when the artifacts are present locally (models/runtime_audit/), their sha256
    match the lock and they pass the strict audit;
  * the ExecuTorch export of the streaming step matches PyTorch (skipped without executorch).

  python3 training/tests/test_runtime_privacy.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
AUDIT = os.path.join(ROOT, "scripts", "audit_native_runtime.py")
LOCK = os.path.join(TRAINING, "android", "runtime_lock.json")
LOCAL = os.path.join(ROOT, "models", "runtime_audit")


def have(tool):
    return shutil.which(tool) is not None


def compile_so(tmp, name, code):
    c = os.path.join(tmp, name + ".c")
    so = os.path.join(tmp, name + ".so")
    with open(c, "w") as f:
        f.write(code)
    subprocess.run(["gcc", "-shared", "-fPIC", "-O0", "-o", so, c], check=True)
    return so


def aar(tmp, name, so=None, classes=None, manifest=None):
    p = os.path.join(tmp, name + ".aar")
    with zipfile.ZipFile(p, "w") as z:
        if so:
            z.write(so, "jni/arm64-v8a/lib" + name + ".so")
        if classes:
            inner = os.path.join(tmp, name + "_classes.jar")
            with zipfile.ZipFile(inner, "w") as j:
                for cname, blob in classes.items():
                    j.writestr(cname, blob)
            z.write(inner, "classes.jar")
        z.writestr("AndroidManifest.xml", manifest or '<manifest package="x"><application/></manifest>')
    return p


def audit(*paths, extra=()):
    return subprocess.run([sys.executable, AUDIT, "--policy", "strict", *extra, *paths], capture_output=True, text=True)


CLEAN = "float f(float x){return x*2.0f;}\n"
SOCKET = "#include <sys/socket.h>\nint g(void){return socket(2,1,0);}\n"
TELEMETRY = 'const char* k = "https://mobile.events.data.microsoft.com/OneCollector/1.0";\nint h(void){return k[0];}\n'
URL = 'const char* u = "https://collect.example-analytics.net/v1/upload";\nint h(void){return u[0];}\n'


@unittest.skipUnless(have("gcc") and have("nm") and have("readelf"), "needs gcc + binutils")
class StaticAuditor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_clean_library_passes(self):
        r = audit(aar(self.tmp, "clean", compile_so(self.tmp, "clean", CLEAN)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("CLEAN", r.stdout)

    def test_network_import_fails(self):
        r = audit(aar(self.tmp, "net", compile_so(self.tmp, "net", SOCKET)))
        self.assertEqual(r.returncode, 1)
        self.assertIn("network-capable imports", r.stdout)

    def test_telemetry_marker_fails(self):
        r = audit(aar(self.tmp, "tele", compile_so(self.tmp, "tele", TELEMETRY)))
        self.assertEqual(r.returncode, 1)
        self.assertIn("telemetry markers", r.stdout)

    def test_hard_coded_url_fails(self):
        r = audit(compile_so(self.tmp, "url", URL))
        self.assertEqual(r.returncode, 1)
        self.assertIn("hard-coded URLs", r.stdout)

    def test_java_network_classes_fail(self):
        cls = {"com/x/Uploader.class": b"\xca\xfe\xba\xbe....java/net/HttpURLConnection....",
               "com/x/telemetry/Sender.class": b"\xca\xfe\xba\xbe....Telemetry...."}
        r = audit(aar(self.tmp, "java", compile_so(self.tmp, "java", CLEAN), classes=cls))
        self.assertEqual(r.returncode, 1)
        self.assertIn("Java bytecode", r.stdout)

    def test_manifest_network_permission_fails(self):
        man = ('<manifest package="x"><uses-permission android:name="android.permission.INTERNET"/>'
               '<application/></manifest>')
        r = audit(aar(self.tmp, "perm", compile_so(self.tmp, "perm", CLEAN), manifest=man))
        self.assertEqual(r.returncode, 1)
        self.assertIn("android.permission.INTERNET", r.stdout)
        # binary (UTF-16) manifest as in an APK, with --skip-java
        p = os.path.join(self.tmp, "app.apk")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("AndroidManifest.xml", b"\x03\x00\x08\x00" + "android.permission.INTERNET".encode("utf-16-le"))
            z.write(compile_so(self.tmp, "app", CLEAN), "lib/arm64-v8a/libapp.so")
        self.assertEqual(audit(p, extra=("--skip-java",)).returncode, 1)


class PinnedRuntime(unittest.TestCase):
    def test_lock_pins_executorch_and_rejects_ort_official(self):
        lock = json.load(open(LOCK))
        names = [a["maven"] for a in lock["artifacts"]]
        self.assertTrue(any(n.startswith("org.pytorch:executorch-android:") for n in names))
        self.assertFalse(any("onnxruntime" in n for n in names), "official ORT Android ships 1DS telemetry")
        for a in lock["artifacts"]:
            self.assertRegex(a["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(a["url"].startswith("https://repo1.maven.org/maven2/"))

    def test_local_artifacts_match_lock_and_pass_strict_audit(self):
        lock = json.load(open(LOCK))
        local = {"executorch-android-1.5.1.aar": "executorch-1.5.1.aar"}
        paths = []
        for a in lock["artifacts"]:
            p = os.path.join(LOCAL, local.get(a["file"], a["file"]))
            if not os.path.exists(p):
                self.skipTest(f"{p} not downloaded (CI downloads it)")
            self.assertEqual(hashlib.sha256(open(p, "rb").read()).hexdigest(), a["sha256"], p)
            paths.append(p)
        r = audit(*paths)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_app_has_no_inference_runtime_or_network_permission(self):
        """Android integration is forbidden before READY: the app must not depend on any
        inference runtime yet, and its manifest must not request network permissions."""
        gradle = open(os.path.join(ROOT, "app", "build.gradle.kts" if os.path.exists(
            os.path.join(ROOT, "app", "build.gradle.kts")) else "build.gradle")).read()
        for dep in ("onnxruntime", "executorch", "tensorflow-lite", "litert", "ncnn"):
            self.assertNotIn(dep, gradle)
        import re
        man = open(os.path.join(ROOT, "app", "src", "main", "AndroidManifest.xml")).read()
        man = re.sub(r"<!--.*?-->", "", man, flags=re.S)
        self.assertNotIn("android.permission.INTERNET", man)
        self.assertNotIn("ACCESS_NETWORK_STATE", man)


def _executorch():
    try:
        import executorch  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_executorch(), "executorch not installed (pip install executorch==1.5.1)")
class ExecuTorchExport(unittest.TestCase):
    def test_fp32_and_int8_streaming_parity(self):
        import numpy as np
        import torch
        from executorch.runtime import Runtime
        from export.export_executorch import build
        from models.stream_anon import StreamAnon, StreamAnonConfig
        torch.manual_seed(0)
        m = StreamAnon(StreamAnonConfig()).eval()
        g = torch.Generator().manual_seed(1)
        T = 40
        mel, pros, spk = torch.randn(1, T, 80, generator=g), torch.randn(1, T, 3, generator=g), torch.randn(1, 128, generator=g)
        with torch.no_grad():
            ref, _ = m(mel, pros, spk)
        tmp = tempfile.mkdtemp()
        try:
            for int8, tol in ((False, 1e-4), (True, 0.35)):
                prog = build(m, frames=2, int8=int8)
                p = os.path.join(tmp, f"s{int(int8)}.pte")
                open(p, "wb").write(prog.buffer)
                meth = Runtime.get().load_program(p).load_method("forward")
                state, outs = m.initial_state(1), []
                for k in range(0, T, 2):
                    o = meth.execute([mel[:, k:k + 2], pros[:, k:k + 2], spk, *state])
                    outs.append(o[0])
                    state = list(o[1:])
                y = torch.cat(outs, 1)
                rel = float((y - ref).norm() / ref.norm())
                self.assertTrue(bool(torch.isfinite(y).all()))
                self.assertLess(rel, tol, f"int8={int8} rel err {rel}")
                if int8:
                    self.assertLess(len(prog.buffer) / 1e6, 30.0)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
