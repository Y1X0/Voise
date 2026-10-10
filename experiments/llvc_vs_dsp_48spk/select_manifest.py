# Pre-registration step: select the evaluation sample from google/svq (CC BY 4.0) and print the manifest.
# Runs as a Kaggle kernel (CPU). No model is run; no audio is kept or uploaded.
#
# Design (fixed before any result is seen; see PROTOCOL.md):
#   - 4 Arabic locale groups (ar_eg, ar_x_gulf, ar_x_levant, ar_x_maghrebi), "clean" condition;
#   - target 12 speakers per group (6 female + 6 male when available), 5 clips each, clip >= 1.5 s;
#   - a speaker is eligible only if it appears in exactly one Arabic locale (any condition);
#   - the 12 speakers of the earlier exploratory/audit sample are excluded (fresh hold-out);
#   - no two selected clips share the same prompt text;
#   - any shortfall is reported, never filled with other data.
# Output: manifest printed between MANIFEST_JSON markers (utt_id, speaker, locale, gender, sec, text).

import io, json, random, urllib.request

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

SEED = 20261011
N_SPK, N_CLIPS, MIN_SEC = 12, 5, 1.5
LOCALES = ["ar_eg", "ar_x_gulf", "ar_x_levant", "ar_x_maghrebi"]
CONDS = ["clean", "background_speech", "media_noise", "traffic_noise"]
PRIOR = "https://raw.githubusercontent.com/Y1X0/Voise/49562deac0c773ae64281537d05450e13dbfd270/experiments/llvc_arabic_svq/audit_report.json"

prior_spk = {c["speaker"] for c in json.loads(urllib.request.urlopen(PRIOR).read())["audit"]["per_clip"]}
assert len(prior_spk) == 12, prior_spk

fs = HfFileSystem()
spk_locs = {}
for loc in LOCALES:
    for cond in CONDS:
        pf = pq.ParquetFile(fs.open(f"datasets/google/svq/2.0.0/utts_{loc}_{cond}.parquet", block_size=1 << 20))
        for s in set(pf.read(columns=["speaker_id"]).column("speaker_id").to_pylist()):
            spk_locs.setdefault(s, set()).add(loc)

rng = random.Random(SEED)
stats, manifest, used_text, bytes_read = {}, [], set(), 0
for loc in LOCALES:
    pf = pq.ParquetFile(fs.open(f"datasets/google/svq/2.0.0/utts_{loc}_clean.parquet", block_size=1 << 20))
    m = pf.read(columns=["utt_id", "speaker_id", "speaker_gender", "text"]).to_pydict()
    rg_rows = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    by = {}
    for i, (s, g) in enumerate(zip(m["speaker_id"], m["speaker_gender"])):
        by.setdefault(s, dict(g=g, rows=[]))["rows"].append(i)
    elig = sorted(s for s, v in by.items() if len(spk_locs[s]) == 1 and s not in prior_spk and len(v["rows"]) >= N_CLIPS)
    stats[loc] = dict(speakers_in_file=len(by), multi_locale=sum(len(spk_locs[s]) > 1 for s in by),
                      prior_sample=sum(s in prior_spk for s in by), eligible=len(elig),
                      eligible_by_gender={g: sum(by[s]["g"] == g for s in elig) for g in sorted({by[s]["g"] for s in elig})})
    rng.shuffle(elig)
    order = [s for s in elig if by[s]["g"] == "female"][: N_SPK // 2] + [s for s in elig if by[s]["g"] == "male"][: N_SPK // 2]
    order += [s for s in elig if s not in order]  # fill-up candidates (other gender / other classes), in shuffled order
    # read the waveform column once per row group to get durations
    audio_rg = {}

    def wav(r):
        global bytes_read
        g = int(np.searchsorted(rg_rows, r, side="right") - 1)
        if g not in audio_rg:
            col = pf.schema_arrow.names.index("waveform")
            bytes_read += pf.metadata.row_group(g).column(col).total_compressed_size
            audio_rg[g] = pf.read_row_group(g, columns=["waveform"]).column("waveform").to_pylist()
        w = audio_rg[g][r - rg_rows[g]]
        return w["bytes"] if isinstance(w, dict) else w

    picked = []
    for s in order:
        if len(picked) == N_SPK:
            break
        rows = by[s]["rows"][:]
        rng.shuffle(rows)
        clips = []
        for r in rows:
            if m["text"][r] in used_text:
                continue
            info = sf.info(io.BytesIO(wav(r)))
            sec = info.frames / info.samplerate
            if sec >= MIN_SEC:
                clips.append(dict(utt_id=m["utt_id"][r], speaker=s, locale=loc, gender=by[s]["g"], sec=round(sec, 3), text=m["text"][r]))
            if len(clips) == N_CLIPS:
                break
        if len(clips) == N_CLIPS:
            picked.append(s)
            used_text.update(c["text"] for c in clips)
            manifest += clips
    audio_rg.clear()
    stats[loc]["selected"] = len(picked)
    stats[loc]["selected_by_gender"] = {g: sum(by[s]["g"] == g for s in picked) for g in sorted({by[s]["g"] for s in picked})}
    print(loc, json.dumps(stats[loc]), flush=True)

out = dict(seed=SEED, design=dict(speakers_per_locale=N_SPK, clips_per_speaker=N_CLIPS, min_sec=MIN_SEC, condition="clean",
                                  exclude_multi_locale=True, exclude_prior_sample=True, unique_text=True),
           source="google/svq 2.0.0 (CC BY 4.0); Heigold et al., MSEB",
           stats=stats, n_speakers=len({c["speaker"] for c in manifest}), n_clips=len(manifest),
           audio_bytes_read_compressed=bytes_read, manifest=manifest)
print("MANIFEST_JSON_BEGIN"); print(json.dumps(out, ensure_ascii=False)); print("MANIFEST_JSON_END", flush=True)
