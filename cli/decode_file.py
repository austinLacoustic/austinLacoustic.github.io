# cli/decode_file.py
import os
import numpy as np
import soundfile as sf
from formats.alak_loader import load_alak
from core.decoder import decode_basei_sparse

def _fix_length(y: np.ndarray, n: int) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    if n <= 0:
        return y
    if y.size == n:
        return y
    if y.size > n:
        return y[:n]
    return np.pad(y, (0, n - y.size), mode="constant")

def decode_file(input_path=None, output_path=None):
    try:
        if input_path is None or not input_path.strip():
            input_path = input("📂 Enter .alak/.alak.gz/.alak.bz2/.alak.xz/.alak.zst to decode: ").strip()
        if not input_path:
            print("⚠️ No input path provided.")
            return

        # Default output name if blank: <input_stem>.decoded.wav
        if output_path is None:
            default_wav = os.path.splitext(os.path.basename(input_path))[0] + ".decoded.wav"
            output_path = input(f"💽 Enter output .wav filename [{default_wav}]: ").strip() or default_wav

        print(f"📂 Loading: {input_path}")
        data = load_alak(input_path)  # robust loader for plain/ compressed .alak
        payload = data.get("compressed_data", data)  # inner payload if present
        sr = int(data.get("sample_rate", 44100))

        scheme = payload.get("scheme", "time_intzz_v1")
        orig_len = int(payload.get("original_length", 0))
        print(f"🧩 Scheme: {scheme}   🎶 Sample rate: {sr} Hz   📏 Declared length: {orig_len}")

        # If labeled Base-i but streams are missing, fall back to scalar path by patching the header.
        if scheme == "bi_pair_intzz_v1":
            has_bi = ("real_zz" in payload or "real_zz_rle" in payload) and \
                     ("imag_zz" in payload or "imag_zz_rle" in payload)
            has_scalar = ("signal_zz" in payload or "signal_zz_rle" in payload)
            if not has_bi and has_scalar:
                print("⚠️ Base-i streams absent; falling back to scalar decode.")
                payload = dict(payload)
                payload["scheme"] = "time_intzz_v1"
            elif not has_bi and not has_scalar:
                print("⚠️ No decodable streams found; decoding to silence.")
                sig = np.zeros((max(0, orig_len),), dtype=np.float32)
                sf.write(output_path, sig, sr)
                print(f"✅ Decode complete (silence written) → {output_path}")
                return

        print("🔄 Decoding...")
        signal = decode_basei_sparse(payload)  # np.float32

        # If original_length is present, trim/pad to it
        if orig_len > 0:
            signal = _fix_length(signal, orig_len)

        # Guard against NaN/Inf
        if not np.all(np.isfinite(signal)):
            print("⚠️ Non-finite samples detected; replacing with zeros.")
            signal = np.nan_to_num(signal, copy=False)

        dur = signal.size / float(sr) if sr > 0 else 0.0
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        rms = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
        print(f"🕒 Duration: {dur:.2f} s   🔊 Peak: {peak:.6f}   RMS: {rms:.6f}")

        print(f"💽 Writing WAV: {output_path}")
        sf.write(output_path, signal.astype(np.float32, copy=False), sr)
        print("✅ Decode complete.")
    except Exception as e:
        print(f"❌ Failed to decode .alak file: {e}")
