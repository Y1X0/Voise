"""Interruption-safe checkpoints for free / preemptible GPUs (Kaggle, Colab, spot instances).

* atomic write: serialise to <path>.tmp-<pid>, flush + fsync, os.replace, fsync the directory.
  A kill at any moment leaves either the previous checkpoint or the new one, never a torn file.
* integrity sidecar <path>.json: sha256 of the checkpoint + run metadata (step, stage, seed,
  config / manifest / environment hashes, git commit, GPU). Written atomically AFTER the
  checkpoint, so a checkpoint without a matching sidecar is treated as incomplete.
* rotation: numbered checkpoints step_<n>.pt keep the newest `keep`; tags (last, best, half,
  stage exits) are never rotated.
* latest_valid(dir): newest checkpoint whose sha256 matches its sidecar (corrupt or partial
  ones are skipped), used to resume after a disconnect.
"""
import glob
import hashlib
import json
import os


def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write_bytes(path, writer):
    """writer(fileobj) writes the content; the file appears at `path` atomically."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    _fsync_dir(d)


def atomic_save(obj, path, meta=None):
    import torch
    atomic_write_bytes(path, lambda f: torch.save(obj, f))
    side = dict(meta or {}, checkpoint=os.path.basename(path), sha256=sha256_file(path))
    atomic_write_bytes(path + ".json", lambda f: f.write(json.dumps(side, indent=1, default=str).encode()))
    return side


def is_valid(path):
    side = path + ".json"
    if not (os.path.exists(path) and os.path.exists(side)):
        return False
    try:
        with open(side) as f:
            return json.load(f).get("sha256") == sha256_file(path)
    except (OSError, ValueError):
        return False


def rotate(directory, keep=3):
    steps = sorted(glob.glob(os.path.join(directory, "step_*.pt")),
                   key=lambda p: int(os.path.basename(p)[5:-3]))
    for p in steps[:-keep] if keep > 0 else []:
        for q in (p, p + ".json"):
            if os.path.exists(q):
                os.remove(q)


def latest_valid(directory):
    """Newest valid checkpoint in `directory` (numbered or tagged), by sidecar step then mtime."""
    cands = []
    for p in glob.glob(os.path.join(directory, "*.pt")):
        if is_valid(p):
            with open(p + ".json") as f:
                step = json.load(f).get("step", -1)
            cands.append((step, os.path.getmtime(p), p))
    return max(cands)[2] if cands else None
