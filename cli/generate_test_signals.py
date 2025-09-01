# cli/generate_test_signals.py
import os
import numpy as np
import soundfile as sf
from scipy import signal

# Import encoder / alak IO
from core.encoder import encode_basei_sparse
from formats.alak_io import save_alak_file

DEFAULT_SR = 44100
OUT_DIR = os.path.join("test_signals")

# ---- Size helpers ----
BYTES_PER_SAMPLE = {"float32": 4, "int16": 2}
def _estimate_size_bytes(n_samples, channels, dtype="int16"):
    return n_samples * channels * BYTES_PER_SAMPLE.get(dtype, 2)

def _human_size(n_bytes):
    units = ["B","KB","MB","GB","TB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0

# ---- I/O helpers ----
def _ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def _save_wav(name, y, sr):
    _ensure_outdir()
    ch = 1 if y.ndim == 1 else y.shape[1]
    n_bytes = _estimate_size_bytes(len(y), ch, "int16")
    print(f"About to save WAV: {name}  •  {len(y)/sr:.2f}s @ {sr} Hz  •  ~{_human_size(n_bytes)}")
    if n_bytes > 100 * 1024 * 1024:
        ok = input("File is >100MB. Proceed? [y/N]: ").strip().lower()
        if ok != "y":
            print("Cancelled.")
            return

    # normalize if clipping and write as int16 PCM
    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 1.0:
        y = y / peak
    y16 = np.clip(y, -1.0, 1.0)
    y16 = (y16 * 32767.0).astype(np.int16)

    path = os.path.join(OUT_DIR, name)
    sf.write(path, y16, sr, subtype="PCM_16")
    print(f"✅ Saved: {path}")

def _save_alak(name, y, sr, dp_label):
    _ensure_outdir()
    # Encode with Base‑i Sparse (per‑sample rotation, structural pruning)
    compressed = encode_basei_sparse(y, dp=None if dp_label == "raw" else dp_label)
    path = os.path.join(OUT_DIR, name)
    save_alak_file(
        path=path,
        sample_rate=sr,
        original_length=len(y),
        dp=dp_label,
        compressed_data=compressed
    )
    print(f"✅ Saved: {path}")

# ---- Prompts ----
def _prompt_float(msg, default=None):
    s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if s == "" and default is not None:
        return float(default)
    return float(s)

def _prompt_int(msg, default=None):
    s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if s == "" and default is not None:
        return int(default)
    return int(s)

def _prompt_sr(default=DEFAULT_SR):
    sr = _prompt_int("Sample rate (Hz)", default)
    if not (8000 <= sr <= 192000):
        raise ValueError("Sample rate out of sane range (8k–192k).")
    return sr

def _prompt_duration(default_sec):
    d = _prompt_float("Duration (s)", default_sec)
    if d <= 0:
        raise ValueError("Duration must be > 0.")
    if d > 600:  # 10 minutes
        ok = input("Duration >10 minutes. Proceed? [y/N]: ").strip().lower()
        if ok != "y":
            raise ValueError("Cancelled oversized duration.")
    return d

# ---- Signal generators ----
def sine(freq, dur, sr):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def square(freq, duty, dur, sr):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    duty = np.clip(duty, 0.0, 1.0)
    return signal.square(2 * np.pi * freq * t, duty=duty)

def triangle(freq, dur, sr):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return signal.sawtooth(2 * np.pi * freq * t, width=0.5)

def sawtooth(freq, dur, sr):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return signal.sawtooth(2 * np.pi * freq * t, width=1.0)

def chirp_linear(f0, f1, dur, sr):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return signal.chirp(t, f0=f0, f1=f1, t1=dur, method="linear")

def impulse(dur, sr):
    n = int(sr * dur)
    y = np.zeros(n, dtype=np.float32)
    if n > 0:
        y[0] = 1.0
    return y

def silence(dur, sr):
    return np.zeros(int(sr * dur), dtype=np.float32)

def white_noise(dur, sr, std=0.3):
    return np.random.normal(0.0, std, int(sr * dur))

def brown_noise(dur, sr, std=0.1):
    w = np.random.normal(0.0, std, int(sr * dur))
    b = np.cumsum(w)
    b = b - np.mean(b)
    if np.max(np.abs(b)) > 0:
        b = b / np.max(np.abs(b)) * 0.9
    return b

def pink_noise(dur, sr, num_rows=16):
    n = int(sr * dur)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rows = np.zeros((num_rows, n))
    vals = np.zeros(num_rows)
    for i in range(n):
        r = np.random.rand(num_rows)
        flip = r < (1.0 / (2 ** np.arange(num_rows)))
        vals[flip] = np.random.normal(0.0, 1.0, flip.sum())
        rows[:, i] = vals
    y = rows.sum(axis=0) / num_rows
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.9
    return y

# ---- Output format / precision selection ----
def _select_output_and_save(base_name, y, sr):
    print("\nSelect Output Format / Precision:")
    print("[0] raw basei_sparse  → .alak")
    print("[5] .wav output       → .wav (16-bit PCM)")
    print("[4] dp4 basei_sparse  → .alak")
    print("[3] dp3 basei_sparse  → .alak")
    print("[2] dp2 basei_sparse  → .alak")
    print("[1] dp1 basei_sparse  → .alak")
    sel = input("Enter selection [0,1,2,3,4,5]: ").strip()

    dp_map = {
        "0": "raw",
        "1": "dp1",
        "2": "dp2",
        "3": "dp3",
        "4": "dp4",
        "5": "wav"
    }
    choice = dp_map.get(sel)
    if choice is None:
        print("❌ Invalid selection.")
        return

    if choice == "wav":
        fname = f"{base_name}.wav"
        _save_wav(fname, y, sr)
    else:
        # .alak save via Base‑i Sparse
        fname = f"{base_name}.{choice}.alak" if choice != "raw" else f"{base_name}.raw.alak"
        _save_alak(fname, y, sr, dp_label=choice)

# ---- Main interactive menu ----
def generate_signal_menu():
    print("\nSelect Signal Type:\n")
    print("[0] Sine → frequency, duration")
    print("[1] Square → frequency, duty cycle, duration")
    print("[2] Triangle → frequency, duration")
    print("[3] Sawtooth → frequency, duration")
    print("[4] Chirp → start freq, end freq, duration")
    print("[5] Impulse or Click → duration (impulse at t=0)")
    print("[6] Silence → duration")
    print("[7] White Noise → duration")
    print("[8] Brown Noise → duration")
    print("[9] Pink Noise → duration")

    try:
        choice = _prompt_int("Enter selection [0–9]")
    except Exception:
        print("❌ Invalid selection.")
        return

    try:
        sr = _prompt_sr(DEFAULT_SR)
    except Exception as e:
        print(f"❌ {e}")
        return

    try:
        if choice == 0:
            f = _prompt_float("Frequency (Hz)", 1000)
            d = _prompt_duration(3.0)
            y = sine(f, d, sr)
            base = f"sine_{int(f)}Hz_{d:.2f}s"

        elif choice == 1:
            f = _prompt_float("Frequency (Hz)", 250)
            duty = _prompt_float("Duty cycle (0.0–1.0)", 0.5)
            d = _prompt_duration(3.0)
            y = square(f, duty, d, sr)
            base = f"square_{int(f)}Hz_d{duty:.2f}_{d:.2f}s"

        elif choice == 2:
            f = _prompt_float("Frequency (Hz)", 500)
            d = _prompt_duration(3.0)
            y = triangle(f, d, sr)
            base = f"triangle_{int(f)}Hz_{d:.2f}s"

        elif choice == 3:
            f = _prompt_float("Frequency (Hz)", 500)
            d = _prompt_duration(3.0)
            y = sawtooth(f, d, sr)
            base = f"sawtooth_{int(f)}Hz_{d:.2f}s"

        elif choice == 4:
            f0 = _prompt_float("Start frequency (Hz)", 100.0)
            f1 = _prompt_float("End frequency (Hz)", 5000.0)
            d = _prompt_duration(5.0)
            y = chirp_linear(f0, f1, d, sr)
            base = f"chirp_{int(f0)}to{int(f1)}Hz_{d:.2f}s"

        elif choice == 5:
            d = _prompt_duration(1.0)
            y = impulse(d, sr)
            base = f"impulse_{d:.2f}s"

        elif choice == 6:
            d = _prompt_duration(3.0)
            y = silence(d, sr)
            base = f"silence_{d:.2f}s"

        elif choice == 7:
            d = _prompt_duration(3.0)
            y = white_noise(d, sr)
            base = f"white_noise_{d:.2f}s"

        elif choice == 8:
            d = _prompt_duration(3.0)
            y = brown_noise(d, sr)
            base = f"brown_noise_{d:.2f}s"

        elif choice == 9:
            d = _prompt_duration(3.0)
            y = pink_noise(d, sr)
            base = f"pink_noise_{d:.2f}s"

        else:
            print("❌ Invalid selection.")
            return

    except Exception as e:
        print(f"❌ {e}")
        return

    # Let the user choose amplitude/DC for any waveform
    try:
        amp = _prompt_float("Amplitude (0.0–1.0)", 1.0)
        dc  = _prompt_float("DC offset (-0.5..0.5)", 0.0)
    except Exception:
        amp, dc = 1.0, 0.0

    # Apply and keep it safe
    y = np.clip(amp * y + dc, -1.0, 1.0)

    # Make filenames self-describing
    if abs(amp - 1.0) > 1e-9:
        base += f"_amp{amp:.5f}"
    if abs(dc) > 1e-9:
        base += f"_dc{dc:+.3f}"

    _select_output_and_save(base, y, sr)

if __name__ == "__main__":
    generate_signal_menu()
