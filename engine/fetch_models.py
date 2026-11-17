"""Download the engine's models at image build time, pinned to exact revisions."""
import os
import sys

from huggingface_hub import snapshot_download

DST = sys.argv[1] if len(sys.argv) > 1 else "/models"
PINS = [
    ("oddadmix/lahgtna-omnivoice-v2", "55c38a613e31", "lahgtna-omnivoice-v2", ["*.json", "*.safetensors", "*.jinja"]),
    ("eustlb/higgs-audio-v2-tokenizer", "528e871c2a26", "lahgtna-omnivoice-v2/audio_tokenizer", ["*.json", "*.safetensors"]),
    ("openai/whisper-large-v3-turbo", "41f01f3fe87f", "whisper-large-v3-turbo", ["*.json", "*.safetensors", "*.txt"]),
]
for repo, rev, sub, patterns in PINS:
    path = snapshot_download(repo, revision=rev, local_dir=os.path.join(DST, sub), allow_patterns=patterns)
    print(repo, rev, "->", path, flush=True)
