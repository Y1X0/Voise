# Metadata-only speaker census of the Arabic varieties in facebook/omnilingual-asr-corpus. Kaggle CPU kernel.
# Reads ONLY these parquet columns: language, iso_639_3, glottocode, speaker_id, prompt_id, segment_id, duration.
# The `audio` column (and `raw_text`) are never read. Output is aggregate counts only: no speaker IDs are printed.
#
# Usable-speaker rule (fixed before running; supports c = 10 clips of 4-8 s with at most 2 clips per segment
# drawn from >= 3 prompts, as in arabic_eval_v2/PROPOSAL.md):
#   PRIMARY  : >= 5 segments AND >= 3 distinct prompt_id AND >= 120 s total duration (all splits pooled)
#   SENSITIVITY rules (reported, not used for the decision):
#     LOOSE  : >= 3 segments, >= 2 prompts, >= 60 s
#     STRICT : >= 10 segments, >= 5 prompts, >= 300 s
# Speaker key = (config, speaker_id). The same speaker_id string in two configs is NOT assumed to be one person.
import collections, json
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "facebook/omnilingual-asr-corpus"
ARA = {"aao", "abh", "abv", "acm", "acq", "acw", "acx", "acy", "adf", "aeb", "aec", "afb", "ajp", "apc", "apd", "arb", "arq", "ars",
       "ary", "arz", "auz", "avl", "ayh", "ayl", "ayn", "ayp", "pga", "shu", "ssh"}  # ISO 639-3 members of macrolanguage [ara]
COLS = ["language", "iso_639_3", "glottocode", "speaker_id", "prompt_id", "segment_id", "duration"]
RULES = {"PRIMARY": (5, 3, 120.0), "LOOSE": (3, 2, 60.0), "STRICT": (10, 5, 300.0)}

info = HfApi().dataset_info(REPO)
fs = HfFileSystem()
cfgs = sorted(p.split("/")[-1] for p in fs.ls(f"datasets/{REPO}/data", detail=False) if p.split("/")[-1].split("_")[0] in ARA)
out = dict(repo=REPO, revision=info.sha, last_modified=str(info.last_modified), card_license=(info.card_data or {}).get("license"),
           columns_read=COLS, columns_never_read=["audio", "raw_text", "prompt"], configs=cfgs, rules=RULES, per_config={}, bytes_footer_note=
           "parquet column projection: only the listed column chunks and footers are fetched")
tot = collections.Counter()
all_keys = {r: 0 for r in RULES}
for cfg in cfgs:
    files = sorted(p for p in fs.ls(f"datasets/{REPO}/data/{cfg}", detail=False) if p.endswith(".parquet"))
    rows, spk = [], collections.defaultdict(lambda: dict(seg=set(), prompts=set(), dur=0.0, splits=set()))
    schema_ok = True
    for f in files:
        split = f.split("/")[-1].split("-")[0]
        pf = pq.ParquetFile(fs.open(f, block_size=1 << 16))
        names = pf.schema_arrow.names
        cols = [c for c in COLS if c in names]
        schema_ok &= all(c in names for c in ("speaker_id", "prompt_id", "segment_id", "duration"))
        t = pf.read(columns=cols).to_pydict()
        for i in range(len(t["speaker_id"])):
            r = {c: t[c][i] for c in cols}; r["split"] = split; rows.append(r)
    c = collections.Counter(); keys = collections.Counter()
    for r in rows:
        s = r.get("speaker_id")
        c["rows"] += 1; c[f"rows_{r['split']}"] += 1
        if s in (None, ""):
            c["missing_speaker_id"] += 1; continue
        if r.get("duration") is None:
            c["missing_duration"] += 1
        k = (r.get("prompt_id"), r.get("segment_id"), s)
        keys[k] += 1
        d = spk[s]; d["seg"].add((r.get("prompt_id"), r.get("segment_id"))); d["prompts"].add(r.get("prompt_id"))
        d["dur"] += r.get("duration") or 0.0; d["splits"].add(r["split"])
    split_sets = collections.Counter(tuple(sorted(v["splits"])) for v in spk.values())
    durs = sorted(v["dur"] for v in spk.values())
    res = dict(rows=c["rows"], rows_by_split={k[5:]: v for k, v in c.items() if k.startswith("rows_")},
               missing_speaker_id=c["missing_speaker_id"], missing_duration=c["missing_duration"], schema_has_required=schema_ok,
               distinct_speaker_ids=len(spk), duplicate_prompt_segment_speaker_rows=sum(v - 1 for v in keys.values() if v > 1),
               speakers_by_split_membership={"+".join(k): v for k, v in split_sets.items()},
               speakers_in_more_than_one_split=sum(1 for v in spk.values() if len(v["splits"]) > 1),
               language_values=sorted({r.get("language") for r in rows if r.get("language")}),
               glottocodes=sorted({r.get("glottocode") for r in rows if r.get("glottocode")}),
               total_hours=round(sum(durs) / 3600, 2),
               per_speaker_minutes=dict(min=round(durs[0] / 60, 1) if durs else None, median=round(durs[len(durs) // 2] / 60, 1) if durs else None,
                                        max=round(durs[-1] / 60, 1) if durs else None),
               usable={r: sum(1 for v in spk.values() if len(v["seg"]) >= a and len(v["prompts"]) >= b and v["dur"] >= m)
                       for r, (a, b, m) in RULES.items()})
    for r in RULES:
        all_keys[r] += res["usable"][r]
    out["per_config"][cfg] = res
    print(cfg, json.dumps(res), flush=True)
out["totals"] = dict(configs=len(cfgs), distinct_speaker_keys=sum(v["distinct_speaker_ids"] for v in out["per_config"].values()),
                     usable=all_keys, rows=sum(v["rows"] for v in out["per_config"].values()))
print("CENSUS_JSON_BEGIN"); print(json.dumps(out, ensure_ascii=True, default=str)); print("CENSUS_JSON_END", flush=True)
