# Secondary analysis (post hoc), metadata only: read SVQ `text` and `transcript_truth` for the 225 frozen clips.
# No audio column is read. Output printed as ASCII-escaped JSON between markers (avoids log transport corruption).
import hashlib, json, urllib.request
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

URL = "https://raw.githubusercontent.com/Y1X0/Voise/f1c6c7aacbee54c912ba9f82284df13f7e9cc456/experiments/llvc_vs_dsp_48spk/manifest.json"
raw = urllib.request.urlopen(URL).read()
assert hashlib.sha256(raw).hexdigest() == "9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685"
ids = {c["utt_id"]: c["locale"] for c in json.loads(raw)["manifest"]}
fs = HfFileSystem()
rows = {}
for loc in sorted(set(ids.values())):
    pf = pq.ParquetFile(fs.open(f"datasets/google/svq/2.0.0/utts_{loc}_clean.parquet", block_size=1 << 20))
    cols = [c for c in ("utt_id", "text", "transcript_truth", "transcriptions/speech") if c in pf.schema_arrow.names]
    t = pf.read(columns=cols).to_pydict()
    for i, u in enumerate(t["utt_id"]):
        if u in ids:
            rows[u] = {c: t[c][i] for c in cols if c != "utt_id"}
print("found", len(rows), "of", len(ids), flush=True)
print("TT_JSON_BEGIN"); print(json.dumps(rows, ensure_ascii=True)); print("TT_JSON_END", flush=True)
