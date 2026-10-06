"""ctypes binding to the app's C++ pitch tracker (YIN) and noise suppressor (dsp/, unchanged).

The shared library is compiled on first use into training/native/build/ (git-ignored):
  g++ -O2 -std=c++17 -shared -fPIC -Idsp/include dsp/src/{pitch_tracker,biquad,noise_suppressor,fft}.cpp voiceanon_capi.cpp
"""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SOURCES = [os.path.join(ROOT, "dsp", "src", f) for f in
           ("pitch_tracker.cpp", "biquad.cpp", "noise_suppressor.cpp", "fft.cpp")] + [os.path.join(HERE, "voiceanon_capi.cpp")]


def _lib_path():
    h = hashlib.sha256()
    for s in SOURCES + [os.path.join(ROOT, "dsp", "include", "voiceanon", f) for f in
                        ("pitch_tracker.h", "noise_suppressor.h", "biquad.h", "fft.h")]:
        h.update(open(s, "rb").read())
    return os.path.join(HERE, "build", f"libvoiceanon_capi_{h.hexdigest()[:12]}.so")


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        path = _lib_path()
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-I", os.path.join(ROOT, "dsp", "include"),
                            *SOURCES, "-o", path], check=True)
        _LIB = ctypes.CDLL(path)
        fp = ctypes.POINTER(ctypes.c_float)
        _LIB.va_yin_track.argtypes = [fp, ctypes.c_int, ctypes.c_double, ctypes.c_int, fp, fp]
        _LIB.va_noise_suppress.argtypes = [fp, ctypes.c_int, ctypes.c_double, ctypes.c_float, fp]
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def yin_track(x: np.ndarray, sr: int = 16000, hop: int = 160):
    """Runtime-identical F0 (Hz, 0 = unvoiced) and confidence per hop."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(x) // hop
    f0 = np.zeros(n, np.float32)
    conf = np.zeros(n, np.float32)
    lib().va_yin_track(_ptr(x), len(x), float(sr), hop, _ptr(f0), _ptr(conf))
    return f0, conf


def noise_suppress(x: np.ndarray, sr: int = 16000, amount: float = 0.5):
    """Runtime-identical noise suppression; returns (output, latency_samples)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    lat = lib().va_noise_suppress(_ptr(x), len(x), float(sr), float(amount), _ptr(out))
    return out, lat
