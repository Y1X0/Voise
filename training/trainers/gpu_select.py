"""GPU inventory and single-GPU selection for real training (no GPU model is assumed).

Requirement: one CUDA device of the 16 GB class. "16 GB" cards report less than 16e9 usable bytes
(an NVIDIA T4 16 GB reports about 15.8e9), so the threshold is MIN_VRAM_BYTES = 15.0e9 total bytes.
Whether the configured batch really fits is NOT inferred from this number: it is measured by
`trainers/kaggle.py smoke` (peak memory of real training steps of every stage at the profile batch).
Nothing here changes the batch size, the precision or the model to make a smaller GPU fit; a device
below the threshold is refused with a message.
"""

MIN_VRAM_BYTES = 15_000_000_000


def inventory(torch=None):
    """-> [{index, name, total_bytes, total_gb, capability, bf16}] for every visible CUDA device."""
    if torch is None:
        import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        out.append({"index": i, "name": p.name, "total_bytes": int(p.total_memory),
                    "total_gb": round(p.total_memory / 1e9, 2), "capability": f"sm_{cap[0]}{cap[1]}",
                    "bf16": cap[0] >= 8})
    return out


def select(min_bytes=MIN_VRAM_BYTES, torch=None):
    """-> (index or None, inventory, message). The first device meeting the requirement is used;
    a second GPU is never required (multi-GPU is not used by the trainer)."""
    inv = inventory(torch)
    if not inv:
        return None, inv, "GPU: no CUDA GPU visible (training on CPU is not supported)"
    for g in inv:
        if g["total_bytes"] >= min_bytes:
            return g["index"], inv, f"GPU: using cuda:{g['index']} {g['name']} ({g['total_gb']} GB)"
    found = ", ".join(f"cuda:{g['index']} {g['name']} {g['total_gb']} GB" for g in inv)
    return None, inv, (f"GPU: no CUDA device with >= {min_bytes / 1e9:.1f}e9 bytes VRAM (16 GB class) found: {found}. "
                       "Training is refused; the batch size is not reduced automatically.")
