#!/usr/bin/env python3
"""Render the project corpus with VoicePrivacy 2024 baseline B3 (STTTS: phonetic ASR ->
prosody with random per-phone offsets -> GAN-generated ARTIFICIAL speaker embedding ->
FastSpeech2 + HiFi-GAN). The target voice is a GAN sample, not a real person.

Runs the unmodified VPC 2024 pipeline code (GPL-3.0) from a local clone with the
official models (DigitalPhonetics/speaker-anonymization release v2.0) unpacked in
<vpc>/exp/sttts_models. Nothing is downloaded (the VPC install/download shell steps are
skipped; download_precomputed_intermediate_repr is false).

Sessions A/B/C = seeds 101/202/303. Every session gets its own intermediate directory,
so its own 5000 GAN vectors are generated from its seed (VPC default: utterance-level
anonymization, a new artificial speaker per utterance).

Usage (from the repo root, inside the B3 venv: speechbrain 0.5.16, espnet 202310):
  models/venv_b3/bin/python scripts/model_adapters/vpc_b3_render.py \
      --vpc models/vpc2024 --corpus eval-corpus --out eval-b3
then score with scripts/neural_anonymization_eval.py using
  "vpc_b3=cmd:python3 scripts/model_adapters/prerendered.py --root eval-b3 --in {in} --out {out} --seed {seed}"
"""
import argparse
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSIONS = {"A": 101, "B": 202, "C": 303}


def gender(spk):
    s = spk.lower()
    if s.startswith("ami-f") or s.endswith("-female"):
        return "f"
    if s.startswith("ami-m") or s.endswith("-male"):
        return "m"
    return "u"  # metadata only: B3's GAN selection does not use gender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vpc", required=True)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "eval-corpus"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", default="A,B,C")
    ap.add_argument("--silero", default=os.path.join(ROOT, "models", "silero-vad"),
                    help="local clone of github.com/snakers4/silero-vad (tag v5.1.2)")
    a = ap.parse_args()
    a.silero = os.path.abspath(a.silero)
    vpc, corpus, out = (os.path.abspath(p) for p in (a.vpc, a.corpus, a.out))
    os.chdir(vpc)
    sys.path.insert(0, vpc)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # The VPC/IMS-Toucan code predates torch 2.6 and stores numpy objects in its .pt files.
    # Restore the pre-2.6 torch.load default in THIS process only; the files loaded are the
    # official release checkpoints and files this pipeline writes itself.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import numpy as np
    import soundfile as sf
    import torch
    from hyperpyyaml import load_hyperpyyaml
    from utils.data_io import _transform_paths

    # VPC's IMS-Toucan code loads Silero VAD with torch.hub.load('snakers4/silero-vad'),
    # which would contact GitHub at run time. Serve it from the local official clone
    # (snakers4/silero-vad v5.1.2, MIT) instead; nothing else is changed.
    silero = os.path.abspath(a.silero)
    hub_load = torch.hub.load

    def local_hub_load(repo_or_dir, *args, **kw):
        if repo_or_dir == "snakers4/silero-vad":
            kw.pop("force_reload", None)
            return hub_load(silero, *args, source="local", **kw)
        return hub_load(repo_or_dir, *args, **kw)

    torch.hub.load = local_hub_load

    # torchaudio >= 2.9 delegates load() to torchcodec (not installed). Equivalent loader:
    # float32 tensor [channels, samples] in [-1, 1] plus the sample rate, as before.
    import torchaudio

    def sf_load(path, *args, **kw):
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(x.T)), sr

    torchaudio.load = sf_load
    from anonymization.pipelines.sttts import STTTSPipeline

    utts = sorted(f[:-4] for f in os.listdir(corpus) if f.endswith(".wav"))
    timing = {}
    for sess in a.sessions.split(","):
        seed = SESSIONS[sess]
        work = os.path.join(out, "_work", sess)
        data = os.path.join(work, "data", "voise")
        os.makedirs(data, exist_ok=True)
        os.makedirs(os.path.join(work, "intermediate"), exist_ok=True)
        spk = {u: u.split("__")[0] for u in utts}
        with open(os.path.join(data, "wav.scp"), "w") as f:
            f.writelines(f"{u} {corpus}/{u}.wav\n" for u in utts)
        with open(os.path.join(data, "utt2spk"), "w") as f:
            f.writelines(f"{u} {spk[u]}\n" for u in utts)
        with open(os.path.join(data, "spk2utt"), "w") as f:
            for s in sorted(set(spk.values())):
                f.write(s + " " + " ".join(u for u in utts if spk[u] == s) + "\n")
        with open(os.path.join(data, "spk2gender"), "w") as f:
            f.writelines(f"{s} {gender(s)}\n" for s in sorted(set(spk.values())))

        y = open("configs/anon_sttts.yaml").read()
        # per-session GAN vector pool (generated from this session's seed)
        y = y.replace("vectors_file: !ref <models_dir>/anonymization/",
                      "vectors_file: !ref <intermediate_dir>/gan_vectors_")
        y = y.replace("anon_level_utt: [IEMOCAP_test, IEMOCAP_dev, libri_dev, libri_test, train-clean-360]",
                      "anon_level_utt: [voise]")
        cfg = load_hyperpyyaml(y, overrides={
            "data_dir": os.path.join(work, "data"),
            "models_dir": os.path.join(vpc, "exp", "sttts_models"),
            "intermediate_dir": os.path.join(work, "intermediate"),
            "download_precomputed_intermediate_repr": False,
        })
        cfg = _transform_paths(cfg)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        t0 = time.time()
        p = STTTSPipeline(config=cfg, force_compute=False, devices=[torch.device("cpu")])
        scp = p.run_anonymization_pipeline({"voise": __import__("pathlib").Path(data)})["voise"]
        timing[sess] = time.time() - t0
        dst = os.path.join(out, sess)
        os.makedirs(dst, exist_ok=True)
        for u in utts:
            x, sr = sf.read(str(scp[u]))
            assert sr == 16000, (u, sr)
            sf.write(os.path.join(dst, u + ".wav"), x.astype(np.float32), sr, subtype="PCM_16")
        audio = sum(sf.info(f"{corpus}/{u}.wav").duration for u in utts)
        print(f"session {sess}: {len(utts)} utterances, {timing[sess]:.0f} s wall "
              f"for {audio:.0f} s audio (RTF {timing[sess] / audio:.2f}, CPU, incl. model load)")


if __name__ == "__main__":
    main()
