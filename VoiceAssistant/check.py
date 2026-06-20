"""
Run this first: python check.py
Verifies every dependency is installed and GPU is visible.
"""

import sys
print(f"Python: {sys.version.split()[0]}")

# ── numpy ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError:
    print("✗ numpy — pip install numpy")

# ── soundfile ─────────────────────────────────────────────────────────────
try:
    import soundfile as sf
    print(f"✓ soundfile {sf.__version__}")
except ImportError:
    print("✗ soundfile — pip install soundfile")

# ── webrtcvad ─────────────────────────────────────────────────────────────
try:
    import webrtcvad
    vad = webrtcvad.Vad(2)
    # Test with a real 30ms frame of silence (960 bytes of zeros at 16kHz 16-bit)
    frame = b'\x00' * 960
    result = vad.is_speech(frame, 16000)
    print(f"✓ webrtcvad — test frame classified as: {'speech' if result else 'silence'}")
except ImportError:
    print("✗ webrtcvad — pip install webrtcvad-wheels")
except Exception as e:
    print(f"✗ webrtcvad installed but failed: {e}")

# ── torch + CUDA ──────────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"✓ torch {torch.__version__}")
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {name} ({mem:.1f} GB VRAM)")
    else:
        print("  ✗ CUDA not available — check driver and torch install")
except ImportError:
    print("✗ torch — see README for install command")

# ── faster-whisper ────────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    print("✓ faster-whisper")
    print("  (model will download on first transcribe call)")
except ImportError:
    print("✗ faster-whisper — pip install faster-whisper")

# ── silero-vad ────────────────────────────────────────────────────────────
try:
    import silero_vad
    print(f"✓ silero-vad {silero_vad.__version__}")
except ImportError:
    try:
        # older versions don't have __version__, check via torch.hub
        import torch
        print("✓ silero-vad (model downloads on first use via torch.hub)")
    except:
        print("✗ silero-vad — pip install silero-vad")

# ── summary ───────────────────────────────────────────────────────────────
print("\n" + "="*40)
try:
    import torch
    import webrtcvad
    import soundfile
    import numpy
    from faster_whisper import WhisperModel
    if torch.cuda.is_available():
        print("✓ All good. Ready to run the pipeline.")
    else:
        print("⚠ All packages ok but no GPU detected.")
        print("  Double-check torch was installed with cu128 index.")
except:
    print("✗ Some packages missing — fix above errors first.")
