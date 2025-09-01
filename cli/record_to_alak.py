# cli/record_to_alak.py
import os, io, time, math, tempfile, shutil
import numpy as np
import soundfile as sf

# optional recording backend
try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    HAS_SD = False

from contextlib import redirect_stdout
from core.encoder import encode_basei_sparse
from core.decoder import decode_basei_sparse
from formats.alak_io import save_alak_file
from evaluation.metrics import evaluate_metrics
from cli.compress_alak_file import compress_alak_path
from core.codec_utils import quantize_to_ints, deltas, zigzag, rle_encode  # reuse

# Optional dp_auto (auto precision)
try:
    from core.dp_auto import dp_auto
    HAS_DPAUTO = True
except Exception:
    HAS_DPAUTO = False


# ---------- Small helpers ----------
def _compress_quiet(path: str, method: str) -> str:
    sink = io.StringIO()
    with redirect_stdout(sink):
        out_path = compress_alak_path(path, method)
    return out_path

def _choose_control_mode() -> str:
    print("\n🎛️ Control")
    print("  [A] Full Auto — hands-off: record → encode → compress")
    print("  [M] Manual Control — tweak SR/DP/prune/RLE/pipeline/compression")
    sel = (input("Select [A/M]: ").strip().upper() or "A")
    return "AUTO" if sel == "A" else "MANUAL"

def _choose_rle_mode() -> bool | None:
    print("\n🧱 RLE (Run-Length Encoding)")
    print("  [A] AUTO – Let encoder decide")
    print("  [0] OFF  – Disable RLE")
    print("  [1] ON   – Force RLE")
    sel = (input("Choose RLE mode [A/0/1]: ").strip().upper() or "A")
    return {"A": None, "0": False, "1": True}[sel]

def _choose_pipeline() -> str:
    print("\n🧩 Encoding pipeline")
    print("  [A] AUTO — choose per content")
    print("  [S] Standard (fast) — time Δ→zigzag; tones/silence/squares/noise")
    print("  [B] Base-i (precision) — complex pairwise; music/speech/transients")
    sel = (input("Select [A/S/B]: ").strip().upper() or "A")
    return {"A": "auto", "S": "time_intzz_v1", "B": "basei_sparse_v1"}.get(sel, "auto")

def _rle_should_use(avg_run_len: float, zero_ratio: float, max_run: int) -> bool:
    if avg_run_len >= 4.0: return True
    if zero_ratio >= 0.30: return True
    if max_run >= 16: return True
    return False

def _run_stats_uint64(u64: np.ndarray):
    if u64.size == 0: return 0.0, 0.0, 0
    runs = 1
    zero_cnt = int(u64[0] == 0)
    cur = int(u64[0]); cur_run = 1; max_run = 1
    for x in u64[1:]:
        x = int(x)
        if x == cur:
            cur_run += 1
        else:
            runs += 1
            if cur_run > max_run: max_run = cur_run
            cur = x; cur_run = 1
        if x == 0: zero_cnt += 1
    if cur_run > max_run: max_run = cur_run
    n = u64.size
    return (n / runs), (zero_cnt / n), max_run

def _auto_pipeline(signal: np.ndarray, sr: int, is_raw_dp: bool) -> str:
    if is_raw_dp:
        return "time_intzz_v1"
    # simple spectral flatness-ish check (same as your encode_file)
    from numpy.fft import rfft
    head = signal[:min(len(signal), sr)]
    spec = np.abs(rfft(head)) + 1e-9
    use_basei = (np.var(np.log(spec)) < 2.0)
    return "basei_sparse_v1" if use_basei else "time_intzz_v1"

def _auto_prune_counts(signal: np.ndarray, dp_level: str | None, target_snr_db: float | None) -> int:
    """
    Heuristic: tie pruning to expected dp/SNR and sparsity of quantized residuals.
    - dp4 / target >= ~75 dB → keep detail → p=0
    - dp3 / ~68–75 → p=0 or 1
    - dp2 / ~60–68 → p=1–2
    - dp1 / <60 → p=2–3
    Then refine by zero ratio after quantization (more zeros → can prune a little more).
    """
    if dp_level in (None, "raw"):
        return 0

    # dp → nominal SNR target
    target_map = {"dp4": 75.0, "dp3": 68.0, "dp2": 62.0, "dp1": 55.0}
    t = float(target_snr_db if target_snr_db is not None else target_map.get(dp_level, 68.0))

    # quantize to ints and inspect zeros
    q, _scale = quantize_to_ints(signal, dp_level)
    q = q.astype(np.int32, copy=False)
    zeros = float(np.mean(q == 0.0)) if q.size else 1.0

    # base suggestion by dp/target
    if t >= 74.0:
        base = 0
    elif t >= 66.0:
        base = 0 if zeros < 0.25 else 1
    elif t >= 60.0:
        base = 1 if zeros < 0.30 else 2
    else:
        base = 2 if zeros < 0.35 else 3

    # clamp 0..3 (you can widen if your encoder supports higher)
    return int(max(0, min(3, base)))

def _effective_tags_from_payload(encoded: dict, scheme_internal: str) -> tuple[str, str]:
    rle_bool = bool(encoded.get("rle", False))
    rle_tag = f"rle{1 if rle_bool else 0}"
    if scheme_internal == "basei_sparse_v1":
        on = bool(encoded.get("raclp", False))
        order = int(encoded.get("lp_order") or 0)
    else:
        on = bool(encoded.get("lp", False))
        order = int(encoded.get("lp_order") or 0)
    lp_tag = f"lp{(order if on and order > 0 else 0)}"
    return lp_tag, rle_tag

def _human(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.2f} {u}"
        s /= 1024.0

def _print_sizes(wav_path: str, alak_path: str, cmp_path: str | None):
    base_dir = os.path.dirname(os.path.abspath(alak_path)) or "."
    wav_size  = os.path.getsize(wav_path)
    alak_size = os.path.getsize(alak_path)
    comp_size = os.path.getsize(cmp_path) if cmp_path else None

    print(f"\n📁 Directory: {base_dir}")
    print(f"🎧 Source WAV: {os.path.basename(wav_path)}  ({_human(wav_size)})")
    print(f"📂 Original  : {os.path.basename(alak_path)}  ({_human(alak_size)})")

    if comp_size is not None:
        print(f"📦 Compressed: {os.path.basename(cmp_path)}  ({_human(comp_size)})")
        red1   = (1 - comp_size / alak_size) * 100 if alak_size else 0.0
        ratio1 = (alak_size / comp_size) if comp_size else float("inf")
        frac1  = (comp_size / alak_size) if alak_size else 0.0
        print(f"\n📉 Reduced by (alak→cmp): {red1:.1f}%")
        print(f"📊 Ratio     (alak→cmp): {ratio1:.1f}x smaller")
        print(f"📏 Fraction  (alak→cmp): {frac1:.2f}")

        red2   = (1 - comp_size / wav_size) * 100 if wav_size else 0.0
        ratio2 = (wav_size / comp_size) if comp_size else float("inf")
        frac2  = (comp_size / wav_size) if wav_size else 0.0
        print(f"\n🎯 End-to-End Reduction (wav→cmp): {red2:.1f}%")
        print(f"🎯 End-to-End Ratio     (wav→cmp): {ratio2:.1f}x smaller")
        print(f"🎯 End-to-End Fraction  (wav→cmp): {frac2:.2f}")
    else:
        red   = (1 - alak_size / wav_size) * 100 if wav_size else 0.0
        ratio = (wav_size / alak_size) if alak_size else float("inf")
        frac  = (alak_size / wav_size) if wav_size else 0.0
        print(f"\n📉 Reduction (wav→alak): {red:.1f}%")
        print(f"📊 Ratio     (wav→alak): {ratio:.1f}x smaller")
        print(f"📏 Fraction  (wav→alak): {frac:.2f}")

# ---------- UI helpers ----------
def _list_devices():
    if not HAS_SD:
        print("⚠️ sounddevice not available; using default device.")
        return None
    print("\n🎛️ Available input devices:")
    try:
        for i, info in enumerate(sd.query_devices()):
            if info.get('max_input_channels', 0) > 0:
                name = info.get('name', f"Device {i}")
                print(f"[{i}] {name}")
    except Exception:
        print("⚠️ Could not enumerate devices.")

def _countdown(seconds: int = 3):
    seconds = max(0, int(seconds))
    if seconds == 0:
        return
    print("")  # blank line
    for i in range(seconds, 0, -1):
        print(f"⏳ Recording starts in {i}…", end="\r", flush=True)
        time.sleep(1)
    print("🎙️ GO!                     ")  # clear the line
    try:
        print("\a", end="", flush=True)  # terminal beep if supported
    except Exception:
        pass

def _record(signal_len_sec: float, sample_rate: int, device_idx: int | None, pre_roll_sec: float = 0.0) -> np.ndarray:
    if not HAS_SD:
        raise RuntimeError("Recording requires `sounddevice` (pip install sounddevice).")
    frames_target = int(round(signal_len_sec * sample_rate))
    pre = max(0.0, float(pre_roll_sec))
    frames = frames_target + int(round(pre * sample_rate))

    sd.default.samplerate = sample_rate
    if device_idx is not None:
        sd.default.device = (device_idx, None)

    data = sd.rec(frames, channels=1, dtype="float32")
    sd.wait()
    y = data.reshape(-1)

    # Trim the pre-roll (if any), then pad/trim to exact requested length
    if pre > 0.0:
        y = y[int(round(pre * sample_rate)):]
    if y.size < frames_target:
        pad = np.zeros(frames_target - y.size, dtype=np.float32)
        y = np.concatenate([y, pad])
    else:
        y = y[:frames_target]
    return y

# ---------- Main flow ----------
def record_to_alak(
    duration_seconds: float | None = None,
    output_path: str | None = None,
    dp_level: str | None = None,     # "raw" or "dp1..dp4" (override)
    device_index: int | None = None,
):

    control = _choose_control_mode()

    # Defaults
    sr = 44100
    dur = 5.0
    device_idx = None
    dp_level = None           # "raw" or "dp1..dp4"
    rle_mode = None           # None / False / True
    scheme_internal = "time_intzz_v1"
    level = 1
    prune_counts = 0

    # Optional overrides from caller (keep compatibility with main.py)
    if duration_seconds is not None:
        try:
            dur = float(duration_seconds)
        except Exception:
            pass
    if device_index is not None:
        try:
            device_idx = int(device_index)
        except Exception:
            pass
    if dp_level is not None:
        # Force dp choice if caller provided one (e.g., "raw", "dp1".."dp4")
        # In AUTO flow we'll honor this and skip dp_auto.
        if dp_level.lower() in ("raw", "dp1", "dp2", "dp3", "dp4"):
            forced_dp = dp_level.lower()
        else:
            forced_dp = None
    else:
        forced_dp = None


    # --- Parameters by mode ---
    if control == "AUTO":
        # Keep SR choice simple in AUTO (less friction)
        print("\n🎶 Using default sample rate: 44100 Hz (AUTO)")
        sr = 44100
        dur = float(input("⌛ Enter recording duration seconds [5.0]: ").strip() or "5.0")

        _list_devices()
        raw = input("🎤 Select device index to use (blank = default): ").strip()
        device_idx = int(raw) if raw else None

        # dp_auto, then AUTO prune tied to SNR target
        if HAS_DPAUTO:
            print("🤖 Using dp_auto…")
        else:
            print("⚠️ dp_auto unavailable; using dp4.")

        input("▶️ Ready to start recording? [Enter]")
        # Ask once; default 3 seconds
        cnt = int((input("⏱️ Countdown seconds [3]: ").strip() or "3"))
        # Optional pre-roll (OFF by default; set env ALAK_PREROLL_SEC to enable, e.g., 0.15)
        pre_roll_sec = float(os.environ.get("ALAK_PREROLL_SEC", "0"))
        _countdown(cnt)
        sig = _record(dur, sr, device_idx, pre_roll_sec=pre_roll_sec)
        print("✅ Recording complete.")

        # Normalize prompt (minimal)
        do_norm = (input("\n📈 Normalize amplitude to full range (-1 to 1)? [Y/n]: ").strip().lower() in ("", "y", "yes"))
        if do_norm:
            peak = float(np.max(np.abs(sig))) if sig.size else 0.0
            if peak > 0:
                sig = (sig / peak).astype(np.float32, copy=False)
            print("✅ Normalized signal.")

        # Optional tiny denoise (AUTO)
        deno = (input("🧹 Denoise (high-pass) [A]uto / [y]es / [N]o: ").strip().lower() or "a")
        if deno in ("a", "auto"):
            # super-light check
            if np.mean(np.abs(sig)) < 0.01:
                print("🤖 No significant low noise detected; skipping denoise.")
            else:
                # simple one-pole HPF @ ~20 Hz
                from scipy.signal import butter, filtfilt
                b, a = butter(1, 20.0 / (sr * 0.5), btype="highpass")
                sig = filtfilt(b, a, sig).astype(np.float32)
                print("🤖 Applied light denoise (HPF).")
        elif deno in ("y", "yes"):
            from scipy.signal import butter, filtfilt
            b, a = butter(1, 20.0 / (sr * 0.5), btype="highpass")
            sig = filtfilt(b, a, sig).astype(np.float32)
            print("✅ Denoised (HPF).")

        # Decide dp
        if forced_dp is not None:
            dp_level = forced_dp
            target_snr_map = {"dp1":55.0, "dp2":62.0, "dp3":68.0, "dp4":75.0}
            target_snr = target_snr_map.get(dp_level, 68.0)
            print(f"🧷 DP override from caller → {dp_level}")
        elif HAS_DPAUTO:
            picked = dp_auto(sig, sample_rate=sr)  # 1..4
            dp_level = f"dp{int(picked)}"
            target_snr_map = {1:55.0, 2:62.0, 3:68.0, 4:75.0}
            target_snr = target_snr_map.get(int(picked), 68.0)
            print(f"🤖 dp_auto selected → {dp_level}")
        else:
            dp_level = "dp4"
            target_snr = 75.0


        # RLE AUTO
        rle_mode = None

        # Pipeline AUTO and prune-count AUTO
        scheme_internal = _auto_pipeline(sig, sr, is_raw_dp=(dp_level in (None, "raw")))
        prune_counts = _auto_prune_counts(sig, dp_level=dp_level, target_snr_db=target_snr)

    else:
        # MANUAL
        # Sample rate
        print("\n🎚️ Choose sample rate:")
        sr_table = [
            8000, 11025, 16000, 22050, 32000,
            44100, 48000, 88200, 96000, 176400, 192000
        ]
        for i, v in enumerate(sr_table):
            tag = " (default)" if v == 44100 else ""
            print(f"  [{i}] {v} Hz{tag}")
        print("  [C] Custom…")
        raw = (input("Enter selection [0–10 or C]: ").strip() or "5").upper()
        if raw == "C":
            sr = int(input("Enter custom sample rate: ").strip())
        else:
            idx = max(0, min(10, int(raw)))
            sr = sr_table[idx]

        dur = float(input("⌛ Enter recording duration seconds [5.0]: ").strip() or "5.0")

        _list_devices()
        raw = input("🎤 Select device index to use (blank = default): ").strip()
        device_idx = int(raw) if raw else None

        # DP selection
        print("\n🧠 Choose compression precision:")
        print("  [A] dp_auto (pick dp1–dp4 by target SNR)")
        print("  [0] raw")
        print("  [4] dp4")
        print("  [3] dp3")
        print("  [2] dp2")
        print("  [1] dp1")
        raw = (input("Enter selection [A,0–4]: ").strip() or "A").upper()
        if raw == "A":
            dp_level = None  # decide after recording with dp_auto
        else:
            dp_level = {"0":"raw","4":"dp4","3":"dp3","2":"dp2","1":"dp1"}.get(raw, "dp4")

        # RLE mode (A/0/1)
        rle_mode = _choose_rle_mode()

        # Pipeline
        scheme_choice = _choose_pipeline()

        # Recording
        print("\n🔧 Settings")
        print("===========")
        print(f"🎚️ Device: {device_idx if device_idx is not None else '(default)'}")
        print(f"🎶 Sample rate: {sr} Hz")
        print(f"⌛ Duration: {dur:.2f} s")
        print(f"🧠 Precision: {'auto' if dp_level is None else dp_level}")
        print(f"📦 RLE: { {None:'AUTO', False:'OFF', True:'ON'}[rle_mode] }")
        print(f"🧩 Pipeline: {scheme_choice.upper() if scheme_choice!='auto' else 'AUTO'}")
        print("===========")
        input("▶️ Ready to start recording? [Enter]")
        cnt = int((input("⏱️ Countdown seconds [3]: ").strip() or "3"))
        pre_roll_sec = float(os.environ.get("ALAK_PREROLL_SEC", "0"))
        _countdown(cnt)
        sig = _record(dur, sr, device_idx, pre_roll_sec=pre_roll_sec)
        print("✅ Recording complete.")

        # Normalize / denoise prompts
        if input("\n💽 Save raw .wav recording? [y/N]: ").strip().lower() == "y":
            pass  # will save after normalization step with new naming

        if input("\n📈 Normalize amplitude to full range (-1 to 1)? [y/N]: ").strip().lower() in ("y","yes"):
            peak = float(np.max(np.abs(sig))) if sig.size else 0.0
            if peak > 0:
                sig = (sig / peak).astype(np.float32, copy=False)
            print("✅ Normalized signal.")

        den = (input("🧹 Denoise (high-pass filter for low noise)? [A]UTO / [y]es / [N]o: ").strip().lower() or "a")
        if den in ("a","auto"):
            if np.mean(np.abs(sig)) < 0.01:
                print("🤖 No significant low noise detected; skipping denoise.")
            else:
                from scipy.signal import butter, filtfilt
                b, a = butter(1, 20.0 / (sr * 0.5), btype="highpass")
                sig = filtfilt(b, a, sig).astype(np.float32)
                print("🤖 Applied light denoise (HPF).")
        elif den in ("y","yes"):
            from scipy.signal import butter, filtfilt
            b, a = butter(1, 20.0 / (sr * 0.5), btype="highpass")
            sig = filtfilt(b, a, sig).astype(np.float32)
            print("✅ Denoised (HPF).")

        # dp_auto if selected or apply override
        target_snr = None
        if forced_dp is not None:
            dp_level = forced_dp
            target_snr_map = {"dp1":55.0, "dp2":62.0, "dp3":68.0, "dp4":75.0}
            target_snr = target_snr_map.get(dp_level, 68.0)
            print(f"🧷 DP override from caller → {dp_level}")
        elif dp_level is None:
            if HAS_DPAUTO:
                picked = dp_auto(sig, sample_rate=sr)
                dp_level = f"dp{int(picked)}"
                target_snr_map = {1:55.0, 2:62.0, 3:68.0, 4:75.0}
                target_snr = target_snr_map.get(int(picked), 68.0)
                print(f"🤖 dp_auto selected → {dp_level}")
            else:
                dp_level = "dp4"
                target_snr = 75.0
                print("⚠️ dp_auto unavailable; using dp4.")

        # Pipeline finalize
        if scheme_choice == "auto":
            scheme_internal = _auto_pipeline(sig, sr, is_raw_dp=(dp_level in (None,"raw")))
        else:
            scheme_internal = scheme_choice

        # Prune counts
        raw = input("✂️  Prune tiny quantized values? counts [A=auto, 0..3] (A): ").strip().lower() or "a"
        if raw == "a":
            prune_counts = _auto_prune_counts(sig, dp_level=dp_level, target_snr_db=target_snr)
            print(f"🤖 prune_counts AUTO → {prune_counts}")
        else:
            try:
                prune_counts = max(0, int(raw))
            except Exception:
                prune_counts = 0

    # ---------- Naming convention ----------
    # Raw WAV name (consistent convention): rec.raw.<dur>s_<sr>Hz.wav
    dur_tag = f"{dur:.2f}s"
    wav_name = f"rec.raw.{dur_tag}_{sr}Hz.wav"
    sf.write(wav_name, sig, sr)
    print(f"✅ Raw .wav saved: {wav_name}")

    # Plan tags (before encode)
    dp_tag = dp_level or "raw"
    pipe_tag = "basei" if scheme_internal == "basei_sparse_v1" else "std"

    # Encode
    print(f"\n📦 Encoding using {dp_tag} (prune_counts={prune_counts}, RLE={'AUTO' if rle_mode is None else ('ON' if rle_mode else 'OFF')})...")
    encoded = encode_basei_sparse(
        sig,
        dp=None if dp_level == "raw" else dp_level,
        scheme=scheme_internal,
        use_rle=rle_mode,
        prune_counts=int(prune_counts),
        level=1,
        use_raclp=(scheme_internal == "basei_sparse_v1"),
        use_lp=(scheme_internal == "time_intzz_v1"),
        lp_order=4  # sensible default for each path; autos in encode_file if you want to mirror there
    )

    # Effective tags (AUTO resolved)
    lp_tag, rle_tag = _effective_tags_from_payload(encoded, scheme_internal)
    level_tag = (".L1" if scheme_internal == "basei_sparse_v1" else "")
    base = f"rec.{dp_tag}.{pipe_tag}{level_tag}.{lp_tag}.{rle_tag}.{dur_tag}_{sr}Hz"
    alak_name = base + ".alak"

    if output_path:
        # If a direct filename was provided, use it (ensure .alak suffix)
        if output_path.lower().endswith(".alak"):
            alak_name = output_path
        else:
            alak_name = output_path + ".alak"

    # Save .alak
    sink = io.StringIO()
    with redirect_stdout(sink):
        save_alak_file(
            path=alak_name,
            sample_rate=int(sr),
            original_length=len(sig),
            dp=dp_level,
            compressed_data=encoded,
        )
    print(f"✅ .alak saved: {alak_name}")

    # Metrics
    recon = decode_basei_sparse(encoded)[:len(sig)]
    print("\n📊 Compression Quality Metrics")
    print("================================")
    evaluate_metrics(sig, recon)
    print("================================")

    # Compression: AUTO → quiet bz2; MANUAL → prompt
    cmp_path = None
    if control == "AUTO":
        try:
            cmp_path = _compress_quiet(alak_name, "bz2")
        except Exception as e:
            print(f"❌ Compression failed: {e}")
        _print_sizes(wav_name, alak_name, cmp_path)
    else:
        if input("\n🗜️ Also write a compressed container now? [y/N]: ").strip().lower() == "y":
            print("Choose compression method:")
            print(" [1] bz2 (default)")
            print(" [2] gzip")
            print(" [3] lzma / xz")
            print(" [4] zstd")
            msel = input("Enter method number [1–4]: ").strip()
            method = {"1":"bz2","2":"gzip","3":"lzma","4":"zstd"}.get(msel, "bz2")
            try:
                cmp_path = compress_alak_path(alak_name, method)
            except Exception as e:
                print(f"❌ Compression failed: {e}")
            _print_sizes(wav_name, alak_name, cmp_path)
        else:
            _print_sizes(wav_name, alak_name, None)
