# cli/alak_player.py

import numpy as np
import sounddevice as sd
from formats.alak_loader import load_alak
from core.decoder import decode_basei_sparse

def play_alak(input_path=None, device=None, gain=1.0):
    if input_path is None:
        input_path = input("🎵 Enter .alak/.alak.gz/.bz2/.xz/.zst to play: ").strip()

    data = load_alak(input_path)
    payload = data.get("compressed_data", data)
    sr = int(data.get("sample_rate", 44100))

    signal = decode_basei_sparse(payload)
    signal = (signal.astype(np.float32) * float(gain)).clip(-1.0, 1.0)

    # UX
    dur = len(signal) / sr if sr else 0.0
    print(f"▶️  Playing '{input_path}'  |  {dur:.2f}s @ {sr} Hz  |  gain={gain}")

    # Device select (your existing menu code is fine)
    try:
        sd.default.device = device if device is not None else None
        sd.play(signal, sr)
        sd.wait()
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Playback failed: {e}")
