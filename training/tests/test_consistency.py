"""Internal consistency of the real training path (no download, no training, no dummy artifacts).

Proves, on the REAL configs and the REAL producer code:
  * every path a consumer reads is the path its producer writes (mix -> train.py manifests,
    compute_teacher_units -> feature_cache -> trainer units, train_speaker_encoder -> GpuAssets);
  * speaker-encoder names / architectures are real (ECAPA-TDNN and ResNet-34 are different networks);
  * preflight requires only what train.py consumes (no Whisper checkpoint, no noise/RIR);
  * English-only mix works without Arabic data; the English+Arabic mix fails clearly without it;
  * readiness separates CODE findings from external (GPU / DATA / EXTERNAL_POLICY) blockers;
  * the commands in docs/TRAINING_READINESS_FINAL.md and docs/TRAINING_START_CHECKLIST.md match
    the scripts' arguments and the configs.
Every model built here is a tiny, randomly initialised instance of the real architecture,
created in a temporary directory and deleted; nothing is written to the repository.
"""
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import orchestrate_training as OT                     # noqa: E402
from datasets import build_manifests as BM            # noqa: E402
from datasets import mix as MX                         # noqa: E402
from trainers import requirements as RQ                # noqa: E402

SR = 16000
TRAIN_CFGS = [os.path.join(ROOT, p) for p in OT.TRAIN_CONFIGS]
MIX_EN = os.path.join(ROOT, OT.MIX_CONFIGS["en_only"])
MIX_EN_AR = os.path.join(ROOT, OT.MIX_CONFIGS["en_ar"])
DOCS = [os.path.join(ROOT, "docs", "TRAINING_READINESS_FINAL.md"), os.path.join(ROOT, "docs", "TRAINING_START_CHECKLIST.md")]


def y(p):
    with open(p) as f:
        return yaml.safe_load(f)


def rj(p):
    with open(p) as f:
        return json.load(f)


def voiced(sec, f0, seed):
    t = np.arange(int(sec * SR)) / SR
    rng = np.random.default_rng(seed)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (0.1 * x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


class ProducerConsumerPaths(unittest.TestCase):
    def test_repository_is_internally_consistent(self):
        self.assertEqual(OT.code_problems(), [])

    def test_mix_output_is_the_training_manifest_directory(self):
        for scope, mp in (("en_only", MIX_EN), ("en_ar", MIX_EN_AR)):
            m = y(mp)
            for c in TRAIN_CFGS:
                d = y(c)["data"]
                self.assertEqual(os.path.normpath(m["out"]), os.path.dirname(d["train_manifest"]), (scope, c))
                self.assertEqual(os.path.dirname(d["valid_manifest"]), os.path.dirname(d["train_manifest"]))
        # build_manifests.write (used by mix.py) names the split files exactly as the configs read them
        d = tempfile.mkdtemp()
        try:
            rows = [{"path": f"/x/{s}.flac", "speaker": f"vctk:{s}", "session": "1", "corpus": "vctk", "language": "en",
                     "duration": 3.0, "sr": SR, "split": sp} for s, sp in (("a", "train"), ("b", "valid"))]
            BM.write(rows, d, 1, {}, {})
            cfg = y(TRAIN_CFGS[0])["data"]
            for key in ("train_manifest", "valid_manifest"):
                self.assertTrue(os.path.exists(os.path.join(d, os.path.basename(cfg[key]))), key)
            self.assertTrue(os.path.exists(os.path.join(d, "manifest_index.json")))
        finally:
            shutil.rmtree(d)

    def test_teacher_units_path_matches_producer(self):
        import compute_teacher_units as CTU
        src = open(CTU.__file__).read()
        self.assertIn("os.path.join(out, KMEANS_FILE)", src)          # producer writes <out>/kmeans.npy
        for c in TRAIN_CFGS:
            t = y(c)["data"]["teacher"]
            self.assertEqual(t["units"], os.path.join(t["units_dir"], RQ.KMEANS_FILE))

    def test_inconsistent_configs_are_reported_as_code(self):
        good = y(TRAIN_CFGS[0])
        mixes = [y(MIX_EN), y(MIX_EN_AR)]
        self.assertEqual(RQ.internal_consistency(good, mixes), [])
        cases = {
            "manifests": lambda d: d.update(train_manifest="data/manifests/train.jsonl", valid_manifest="data/manifests/valid.jsonl"),
            "units": lambda d: d["teacher"].update(units="data/teacher/units_k500.npy"),
            "string encoder": lambda d: d.update(speaker_encoders_train=["ecapa_train_inhouse"]),
            "unknown arch": lambda d: d.update(speaker_encoders_train=[{"name": "xvector_a", "arch": "xvector"}]),
            "renamed ecapa": lambda d: d.update(speaker_encoders_train=[{"name": "resnet34_train_inhouse", "arch": "ecapa"}]),
            "augment on": lambda d: d["augment"].update(enabled=True),
            "evaluator as encoder": lambda d: d.update(speaker_encoders_train=[{"name": "ecapa_valid_inhouse", "arch": "ecapa"}]),
        }
        for name, f in cases.items():
            bad = copy.deepcopy(good)
            f(bad["data"])
            self.assertTrue(RQ.internal_consistency(bad, mixes), name)
        bad_mix = dict(mixes[0], out="data/manifests")
        self.assertTrue(RQ.internal_consistency(good, [bad_mix]))


class EndToEndArtifacts(unittest.TestCase):
    """Real producers -> real consumers on a tiny fixture: mix -> teacher units -> feature cache ->
    speaker encoders (ECAPA + ResNet-34) -> train.py preflight -> GpuAssets -> sampler."""

    @classmethod
    def setUpClass(cls):
        from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
        import compute_teacher_units as CTU
        import train_speaker_encoder as TSE
        from datasets import feature_cache as FC
        cls.d = d = tempfile.mkdtemp()
        # tiny random Whisper OUTSIDE models_dir (the teacher is a data-prep input, not a train.py input)
        wc = WhisperConfig(d_model=32, encoder_layers=2, decoder_layers=1, encoder_attention_heads=2,
                           decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80,
                           max_source_positions=1500, max_target_positions=32, vocab_size=64,
                           pad_token_id=1, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
        tdir = os.path.join(d, "prep", "tiny-whisper")
        WhisperModel(wc).save_pretrained(tdir)
        WhisperFeatureExtractor(feature_size=80).save_pretrained(tdir)
        rows = []
        for s in range(6):
            for u in range(2):
                p = os.path.join(d, "wav", f"s{s}_{u}.wav")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, voiced(6.0, 100 + 20 * s, 10 * s + u), SR)
                rows.append({"path": p, "speaker": f"librosa_example:{s}", "session": str(u), "corpus": "librosa_example",
                             "language": "en", "duration": 6.0, "sr": SR, "split": "valid" if s == 5 else "train"})
        cfg = y(TRAIN_CFGS[0])
        dd = cfg["data"]
        rel = lambda p: os.path.join(d, p)                          # the configs' own relative paths, rooted at d
        mdir = rel(os.path.dirname(dd["train_manifest"]))
        BM.write(rows, mdir, 1, {"data_scope": {"name": "en_only_interim", "languages": ["en"]}}, {})
        train_rows = [r for r in rows if r["split"] == "train"]
        n_units = 8
        CTU.run(train_rows, tdir, layer=1, k=n_units, out=rel(dd["teacher"]["units_dir"]))
        FC.build(train_rows, os.path.dirname(rel(dd["train_index"])), rel(dd["teacher"]["units_dir"]))
        FC.build([r for r in rows if r["split"] == "valid"], os.path.dirname(rel(dd["valid_index"])), None)
        cls.models_dir = rel("models_train")
        for spec in dd["speaker_encoders_train"]:
            TSE.train(rel(dd["train_index"]), spec["name"], "TRAIN", 2, rel(f"runs/spk/{spec['name']}"), cls.models_dir,
                      channels=16 if spec["arch"] == "ecapa" else 32, dim=16, batch=2, seg=1.0, licence_path="smoke",
                      device="cpu", arch=spec["arch"], ckpt_every=2)
        for key in ("train_manifest", "valid_manifest", "train_index", "valid_index"):
            dd[key] = rel(dd[key])
        for key in ("units", "units_dir", "unit_teacher"):
            dd["teacher"][key] = rel(dd["teacher"][key])
        dd["augment"]["noise_manifest"] = rel(dd["augment"]["noise_manifest"])
        dd["augment"]["rir_manifest"] = rel(dd["augment"]["rir_manifest"])
        cfg["model"]["n_units"] = n_units
        cls.cfg_raw = cfg
        cls.cfg_path = os.path.join(d, "cfg.yaml")
        with open(cls.cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def cfg(self, raw=None):
        from trainers.config import load_config
        if raw is None:
            return load_config(self.cfg_path)
        p = os.path.join(self.d, "cfg_tmp.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(raw, f)
        return load_config(p)

    def test_preflight_needs_only_what_train_py_consumes(self):
        from trainers.train import preflight
        dd = self.cfg_raw["data"]
        for p in (dd["teacher"]["unit_teacher"], dd["augment"]["noise_manifest"], dd["augment"]["rir_manifest"]):
            self.assertFalse(os.path.exists(p), p)                  # absent on purpose
        self.assertEqual(preflight(self.cfg(), require_gpu=False, models_dir=self.models_dir), [])
        on = copy.deepcopy(self.cfg_raw)
        on["data"]["augment"]["enabled"] = True
        miss = preflight(self.cfg(on), require_gpu=False, models_dir=self.models_dir)
        self.assertTrue(any(m.startswith("CODE:") and "augment" in m for m in miss), miss)

    def test_teacher_output_is_trainer_input(self):
        from datasets.stream_sampler import StreamingSegmentSampler
        dd = self.cfg_raw["data"]
        self.assertEqual(int(np.load(dd["teacher"]["units"]).shape[0]), self.cfg_raw["model"]["n_units"])
        ents = [json.loads(l) for l in open(dd["train_index"])]
        self.assertTrue(ents and all(os.path.dirname(e["units"]) == dd["teacher"]["units_dir"] for e in ents))
        b = StreamingSegmentSampler(dd["train_index"], "train", 1.0, seed=0, licence_path="smoke").batch(0, 2)
        self.assertIn("units", b)
        self.assertLess(int(b["units"].max()), self.cfg_raw["model"]["n_units"])
        wrong = copy.deepcopy(self.cfg_raw)
        wrong["model"]["n_units"] = 500
        self.assertTrue(any("K=8 != model.n_units=500" in m for m in RQ.artifact_requirements(self.cfg(wrong), self.models_dir)))

    def test_speaker_encoders_are_the_declared_architectures(self):
        import train_speaker_encoder as TSE
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.lobes.models.ResNet import BasicBlock, ResNet, SEBasicBlock
        from trainers.assets import GpuAssets
        specs = RQ.encoder_specs(self.cfg_raw["data"])
        self.assertEqual(sorted(s["arch"] for s in specs), ["ecapa", "resnet34"])
        r34 = TSE.Encoder("resnet34", channels=32, dim=16)
        self.assertIsInstance(r34.net, ResNet)
        stages = [r34.net.layer1, r34.net.layer2, r34.net.layer3, r34.net.layer4]
        self.assertEqual([len(s) for s in stages], [3, 4, 6, 3])                        # ResNet-34 layout
        self.assertTrue(all(isinstance(b, (BasicBlock, SEBasicBlock)) for s in stages for b in s))
        self.assertIsInstance(TSE.Encoder("ecapa", channels=16, dim=16).net, ECAPA_TDNN)
        for s in specs:
            meta = json.load(open(os.path.join(self.models_dir, s["name"] + ".json")))
            self.assertEqual((meta["arch"], meta["role"]), (s["arch"], "TRAIN"))
        a = GpuAssets(self.cfg(), device="cpu", models_dir=self.models_dir)
        e = [enc(torch.randn(2, 120, 80)) for enc in a.speaker_encoders]
        self.assertEqual([tuple(x.shape) for x in e], [(2, 16), (2, 16)])
        # a model whose metadata disagrees with the config (renamed / wrong role) is refused
        bad_dir = os.path.join(self.d, "models_bad")
        shutil.copytree(self.models_dir, bad_dir)
        ecapa = next(s["name"] for s in specs if s["arch"] == "ecapa")
        resnet = next(s["name"] for s in specs if s["arch"] == "resnet34")
        for ext in (".pt", ".json"):
            shutil.copy(os.path.join(self.models_dir, ecapa + ext), os.path.join(bad_dir, resnet + ext))
        with self.assertRaises(ValueError):
            GpuAssets(self.cfg(), device="cpu", models_dir=bad_dir)
        self.assertTrue(any("arch/role ecapa/TRAIN != config resnet34/TRAIN" in m
                            for m in RQ.artifact_requirements(self.cfg(), bad_dir)))
        meta = json.load(open(os.path.join(self.models_dir, ecapa + ".json")))
        shutil.copy(os.path.join(self.models_dir, resnet + ".json"), os.path.join(bad_dir, resnet + ".json"))
        json.dump(dict(meta, role="VALID"), open(os.path.join(bad_dir, ecapa + ".json"), "w"))
        with self.assertRaises(ValueError):
            GpuAssets(self.cfg(), device="cpu", models_dir=bad_dir)
        os.remove(os.path.join(bad_dir, ecapa + ".json"))
        with self.assertRaises(FileNotFoundError):
            GpuAssets(self.cfg(), device="cpu", models_dir=bad_dir)

    def test_data_scope_recorded_for_english_only(self):
        from trainers.real_run import data_scope, readiness_scope
        self.assertEqual(data_scope(self.cfg_raw["data"]), {"languages": ["en"], "arabic": "ARABIC_NOT_VERIFIED"})
        self.assertEqual(readiness_scope(self.cfg_raw["data"]), "en_only")


class ArabicMix(unittest.TestCase):
    def setUp(self):
        from datasets import gate as G
        self.d = tempfile.mkdtemp()
        self.G, self.old = G, G.PROVENANCE
        G.PROVENANCE = os.path.join(self.d, "prov.json")
        json.dump({"verified": {"mls_en": {"archive_sha256": "a", "licence_text_sha256": "b"}}}, open(G.PROVENANCE, "w"))
        src = y(MIX_EN)["sources"][0]
        rows = []
        for i, s in enumerate(["1", "2", "3", "4"]):
            for sess in ("1", "2"):
                rows.append({"path": f"/x/{s}/{sess}.flac", "speaker": f"mls_en:{s}", "session": sess, "corpus": "mls_en",
                             "language": "en", "duration": 3.0, "sr": SR, "split": "train" if i < 3 else "test",
                             **({"role": "enroll" if sess == "1" else "trial"} if i == 3 else {})})
        BM.write(rows, os.path.join(self.d, src["manifests"]), 1, {}, {})

    def tearDown(self):
        self.G.PROVENANCE = self.old
        shutil.rmtree(self.d)

    def test_english_only_config_needs_no_arabic(self):
        cfg = y(MIX_EN)
        self.assertEqual(cfg["language_weights"], {"en": 1.0})
        self.assertEqual({s["language"] for s in cfg["sources"]}, {"en"})
        idx, n = MX.mix(cfg, base=self.d)
        self.assertEqual(dict(n), {"en": 6})
        self.assertEqual(idx["inputs"]["data_scope"], {"name": "en_only_interim", "languages": ["en"]})
        self.assertTrue(os.path.exists(os.path.join(self.d, y(TRAIN_CFGS[0])["data"]["train_manifest"])))
        self.assertIn("NOT the project target", open(MIX_EN).read())

    def test_english_arabic_config_fails_clearly_without_arabic(self):
        cfg = y(MIX_EN_AR)
        self.assertEqual(cfg["language_weights"], {"en": 0.7, "ar": 0.3})          # Arabic stays in the target
        ar = [s for s in cfg["sources"] if s["language"] == "ar"]
        self.assertEqual(len(ar), 1)
        self.assertFalse(os.path.exists(os.path.join(self.d, ar[0]["manifests"], "manifest_index.json")))
        with self.assertRaises(MX.MixError) as e:
            MX.mix(cfg, base=self.d)
        self.assertIn("consented Arabic corpus", str(e.exception))
        self.assertIn("arabic_import.py", str(e.exception))
        self.assertFalse(os.path.exists(os.path.join(self.d, cfg["out"])))          # nothing written
        r = subprocess.run([sys.executable, os.path.join(TRAINING, "datasets", "mix.py"), "--config", MIX_EN_AR],
                           capture_output=True, text=True, cwd=self.d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("consented Arabic corpus own_ar_v1 is missing", r.stderr)

    def test_language_weights_must_match_sources(self):
        cfg = y(MIX_EN_AR)
        with self.assertRaises(MX.MixError):                                       # Arabic weight without an Arabic source
            MX.check_config(dict(cfg, sources=[s for s in cfg["sources"] if s["language"] == "en"]))
        with self.assertRaises(MX.MixError):                                       # Arabic source with weight 0
            MX.check_config(dict(cfg, language_weights={"en": 1.0, "ar": 0.0}))
        with self.assertRaises(MX.MixError):
            MX.check_config(dict(cfg, language_weights={"en": 0.7, "ar": 0.2}))


class Readiness(unittest.TestCase):
    def test_no_go_only_for_external_reasons(self):
        r = rj(OT.READINESS)
        self.assertEqual(r["verdict"], "NO-GO")
        self.assertEqual(set(r["conditions"]), set(r["condition_meta"]))
        for k, m in r["condition_meta"].items():
            self.assertIn(m["category"], ("GPU", "DATA", "EXTERNAL_POLICY"), k)
            self.assertTrue(set(m["scopes"]) <= set(OT.SCOPES), k)
        cats = {m["category"] for m in r["condition_meta"].values()}
        self.assertEqual(cats, {"GPU", "DATA", "EXTERNAL_POLICY"})
        probs = OT.preflight()
        self.assertTrue(probs)
        self.assertFalse([p for p in probs if p.startswith("CODE:")], probs)
        for p in probs[1:]:
            self.assertRegex(p, r"^(GPU|DATA|EXTERNAL_POLICY): condition false: ")

    def test_code_failure_is_never_hidden_by_go(self):
        d = tempfile.mkdtemp()
        old = OT.TRAIN_CONFIGS
        try:
            r = rj(OT.READINESS)
            p = os.path.join(d, "ready.json")
            json.dump(dict(r, verdict="GO", conditions={k: True for k in r["conditions"]}), open(p, "w"))
            self.assertEqual(OT.preflight(p), [])
            bad = y(TRAIN_CFGS[0])
            bad["data"]["teacher"]["units"] = "data/teacher/units_k500.npy"
            bp = os.path.join(d, "bad.yaml")
            yaml.safe_dump(bad, open(bp, "w"))
            OT.TRAIN_CONFIGS = (bp,)
            probs = OT.preflight(p)
            self.assertTrue(probs and all(x.startswith("CODE:") for x in probs), probs)
            rep = OT.preflight_report(p)
            self.assertTrue(rep["code"])
            self.assertEqual(rep["external"], [])
        finally:
            OT.TRAIN_CONFIGS = old
            shutil.rmtree(d)

    def test_train_py_lists_only_categorised_external_items(self):
        r = subprocess.run([sys.executable, os.path.join(TRAINING, "trainers", "train.py"), "--config", OT.TRAIN_CONFIGS[0],
                            "--out", os.path.join(tempfile.gettempdir(), "never_written")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 2)
        items = [l.strip()[2:] for l in r.stderr.splitlines() if l.strip().startswith("- ")]
        self.assertTrue(items)
        self.assertEqual(items[0], "verdict is 'NO-GO', not GO")
        for it in items[1:]:
            self.assertRegex(it, r"^(GPU|DATA|EXTERNAL_POLICY): ", it)
            self.assertNotRegex(it, r"noise|rir\.jsonl|^DATA: .*whisper-small$", it)

    def test_scopes(self):
        r = rj(OT.READINESS)
        en = {b["condition"] for b in OT.external_blockers(r, "en_only")}
        ar = {b["condition"] for b in OT.external_blockers(r, "en_ar")}
        self.assertIn("arabic_consented_corpus_available", ar)
        self.assertNotIn("arabic_consented_corpus_available", en)
        self.assertTrue(en < ar)


def doc_commands():
    """Every `python3 …` command inside ```bash blocks of the two docs (continuations joined)."""
    out = []
    for p in DOCS:
        text = open(p, encoding="utf-8").read()
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for line in block.replace("\\\n", " ").splitlines():
                line = line.strip()
                if line.startswith("python3 "):
                    out.append((os.path.basename(p), line))
    return out


def flags(cmd):
    return re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", cmd)


def arg(cmd, name):
    toks = shlex.split(re.sub(r"[\[\]]", " ", cmd))
    return [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == name]


class DocsMatchCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmds = doc_commands()
        cls.help = {}

    def helptext(self, script, sub):
        key = (script, sub)
        if key not in self.help:
            r = subprocess.run([sys.executable, os.path.join(ROOT, script)] + ([sub] if sub else []) + ["-h"],
                               capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(r.returncode, 0, f"{script} {sub} -h: {r.stderr[-500:]}")
            self.help[key] = r.stdout
        return self.help[key]

    def test_both_docs_have_commands_and_no_stale_ones(self):
        names = {n for n, _ in self.cmds}
        self.assertEqual(names, {os.path.basename(p) for p in DOCS})
        for p in DOCS:
            text = open(p, encoding="utf-8").read()
            for stale in ("data_mix.yaml", "units_k500", "data/manifests/train.jsonl", "Config mismatch"):
                self.assertNotIn(stale, text, (p, stale))

    def test_every_flag_exists(self):
        subs = {"mls.py": ("download", "verify", "manifest"), "orchestrate_training.py": ("preflight", "bundle", "plan-resume", "verify"),
                "android_benchmark.py": ("plan", "run-neural", "report")}
        for doc, c in self.cmds:
            toks = shlex.split(re.sub(r"[\[\]<>|]", " ", c))
            script = toks[1]
            self.assertTrue(os.path.exists(os.path.join(ROOT, script)), (doc, c))
            sub = toks[2] if os.path.basename(script) in subs and len(toks) > 2 and not toks[2].startswith("-") else None
            if os.path.basename(script) in subs:
                self.assertIn(sub, subs[os.path.basename(script)], (doc, c))
            h = self.helptext(script, sub)
            for f in flags(c):
                self.assertIn(f, h, (doc, c, f))

    def test_commands_use_config_paths(self):
        cfgs = [y(p) for p in TRAIN_CFGS]
        d = cfgs[0]["data"]
        for c in cfgs[1:]:
            self.assertEqual(c["data"], d)
        mixes = {os.path.relpath(p, ROOT): y(p) for p in (MIX_EN, MIX_EN_AR)}
        seen = {k: set() for k in ("mix", "teacher", "cache", "enc_train", "train", "mls", "arabic")}
        for doc, c in self.cmds:
            if "datasets/mix.py" in c:
                (m,) = arg(c, "--config")
                self.assertIn(m, mixes, (doc, c))
                seen["mix"].add(m)
            elif "compute_teacher_units.py" in c:
                self.assertEqual(arg(c, "--manifest"), [d["train_manifest"]])
                self.assertEqual(arg(c, "--teacher"), [d["teacher"]["unit_teacher"]])
                self.assertEqual(arg(c, "--out"), [d["teacher"]["units_dir"]])
                self.assertEqual(arg(c, "--k"), [str(cfgs[0]["model"]["n_units"])])
                seen["teacher"].add(doc)
            elif "feature_cache.py" in c:
                (cache,) = arg(c, "--cache")
                (man,) = arg(c, "--manifest")
                split = "train" if os.path.join(cache, "index.jsonl") == d["train_index"] else "valid"
                self.assertEqual(os.path.join(cache, "index.jsonl"), d[f"{split}_index"], (doc, c))
                self.assertEqual(man, d[f"{split}_manifest"], (doc, c))
                self.assertEqual(arg(c, "--units-dir"), [d["teacher"]["units_dir"]] if split == "train" else [], (doc, c))
                seen["cache"].add((doc, split))
            elif "train_speaker_encoder.py" in c:
                self.assertEqual(arg(c, "--index"), [d["train_index"]])
                (role,), (name,), (arch,) = arg(c, "--role"), arg(c, "--name"), arg(c, "--arch")
                self.assertEqual(arg(c, "--models-dir") or ["models_train"], ["models_train"])
                if role == "TRAIN":
                    self.assertIn({"name": name, "arch": arch}, d["speaker_encoders_train"], (doc, c))
                    seen["enc_train"].add((doc, name))
                else:
                    held = {n for v in cfgs[0]["evaluators"]["VALID"].values() for n in (v if isinstance(v, list) else [v])}
                    self.assertIn(name, held, (doc, c))
            elif "trainers/train.py" in c:
                (cp,) = arg(c, "--config")
                self.assertIn(cp, OT.TRAIN_CONFIGS)
                (gp,) = arg(c, "--gpu-profile")
                self.assertTrue(os.path.exists(os.path.join(TRAINING, "configs", "gpu", gp + ".yaml")), gp)
                seen["train"].add(doc)
            elif "datasets/mls.py manifest" in c:
                self.assertEqual(arg(c, "--out"), [next(s["manifests"] for s in mixes[OT.MIX_CONFIGS["en_only"]]["sources"])])
                seen["mls"].add(doc)
            elif "arabic_import.py" in c:
                ar = next(s for s in mixes[OT.MIX_CONFIGS["en_ar"]]["sources"] if s["language"] == "ar")
                self.assertEqual(arg(c, "--out"), [os.path.dirname(ar["manifests"])])   # importer writes <out>/manifests
                self.assertEqual(arg(c, "--consents"), [ar["consent_store"]])
                seen["arabic"].add(doc)
        for doc in {os.path.basename(p) for p in DOCS}:
            self.assertEqual({n for dd, n in seen["enc_train"] if dd == doc}, {e["name"] for e in d["speaker_encoders_train"]}, doc)
            self.assertEqual({s for dd, s in seen["cache"] if dd == doc}, {"train", "valid"}, doc)
            for k in ("teacher", "train", "mls", "arabic"):
                self.assertIn(doc, seen[k], (doc, k))
        self.assertEqual(seen["mix"], set(mixes))


if __name__ == "__main__":
    unittest.main()
