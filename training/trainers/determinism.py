"""Run-to-run determinism settings shared by real training and the Kaggle session smoke.

Resume restores every piece of state exactly (weights, optimisers, schedulers, GradScalers, CPU and
CUDA RNG, step counters, abort monitor, history); the data order is a pure function of (seed, step).
What can still differ on a GPU are non-deterministic CUDA kernels. With `deterministic=True` cuDNN
and PyTorch are asked for deterministic algorithms; operations that have none emit a warning
(warn_only) instead of stopping training, and `trainers/kaggle.py smoke` reports which ones did.
"""
import os


def configure(device="cpu", deterministic=True):
    import torch
    if device.startswith("cuda"):
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before the first cuBLAS call
        torch.cuda.set_device(torch.device(device))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {"device": device, "deterministic": bool(deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
