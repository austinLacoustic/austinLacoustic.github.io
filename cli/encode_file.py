# cli/encode_file.py

import os, io, tempfile, uuid, contextlib, shutil, math
import numpy as np
import soundfile as sf

from contextlib import redirect_stdout
from core.encoder import encode_basei_sparse
from core.decoder import decode_basei_sparse
from formats.alak_io import save_alak_file
from evaluation.metrics import evaluate_metrics, evaluate_metrics_basei
from cli.compress_alak_file import compress_alak_path
from core.codec_utils import quantize_to_ints, deltas, zigzag, rle_encode  # for standard LP auto

# Optional dp_auto (auto precision)
try:
    from core.dp_auto import dp_auto
    HAS_DPAUTO = True
except Exception:
    HAS_DPAUTO = False

PIPELINE_MAP = {
    "AUTO": "auto",
    "Standard (fast)": "time_intzz_v1",
    "Base-i (precision)": "basei_sparse_v1",
}

def _compress_quiet(path: str, method: str) -> str:
    """Run compress_alak_path but suppress its internal prints; return output path."""
    sink = io.StringIO()
    with redirect_stdout(sink):
        out_path = compress_alak_path(path, method)
    return out_path

def _save_alak_quiet(path: str, sr: int, sig_len: int, dp_level: str | None, encoded: dict):
    """Save .alak but silence any prints the saver might do."""
    sink = io.StringIO()
    with redirect_stdout(sink):
        save_alak_file(
            path=path,
            sample_rate=int(sr),
            original_length=int(sig_len),
            dp=dp_level,
            compressed_data=encoded,
        )

def _probe_pick_order(
    signal: np.ndarray,
    sr: int,
    dp_level: str | None,
    scheme_internal: str,   # "time_intzz_v1" or "basei_sparse_v1"
    rle_mode: bool | None,  # None(A)/True/False
    level: int,
    orders: list[int],
    basei_mode: bool,       # True => probe RACLP orders, False => standard LP orders
) -> int | None:
    """
    For each p in 'orders', encode->save temp .alak -> .bz2 -> measure size.
    Return the order that produced the smallest .bz2; None if all failed.
    """
    best_p = None
    best_size = None
    tmpdir = tempfile.mkdtemp(prefix="alak_probe_")
    try:
        for p in orders:
            try:
                # Encode with the trial order
                encoded = encode_basei_sparse(
                    signal,
                    dp=None if dp_level == "raw" else dp_level,
                    scheme=scheme_internal,
                    use_rle=rle_mode,
                    prune_counts=0,
                    level=level,
                    use_raclp=basei_mode,
                    use_lp=(not basei_mode),
                    lp_order=p,
                )

                # Save & compress quietly
                alak_path = os.path.join(tmpdir, f"probe_p{p}.alak")
                _save_alak_quiet(alak_path, sr, len(signal), dp_level, encoded)
                try:
                    cmp_path = _compress_quiet(alak_path, "bz2")
                except Exception:
                    # If compression fails, skip this candidate
                    continue

                size = os.path.getsize(cmp_path)
                if (best_size is None) or (size < best_size):
                    best_size = size
                    best_p = p

            except Exception:
                # Any encode/save hiccup → skip this candidate
                continue
    finally:
        # Clean up temp dir
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    return best_p

def _choose_pipeline():
    print("\n🧩 Encoding pipeline")
    print("  [A] AUTO — choose per content")
    print("  [S] Standard (fast) — time Δ→zigzag; great for tones/silence/squares/noise")
    print("  [B] Base-i (precision) — complex pairwise; best for music/speech/transients")
    sel = (input("Select [A/S/B]: ").strip().upper() or "S")
    return {"A": "AUTO", "S": "Standard (fast)", "B": "Base-i (precision)"}.get(sel, "Standard (fast)")

def _rle_should_use(avg_run_len: float, zero_ratio: float, max_run: int) -> bool:
    if avg_run_len >= 4.0: return True
    if zero_ratio >= 0.30: return True
    if max_run >= 16: return True
    return False

def _run_stats_uint64(u64: np.ndarray):
    # simplified stats needed for heuristic
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

def _varint_len_u64(x: int) -> int:
    # LEB128/varint byte count proxy
    if x <= 0: 
        return 1
    # bit_length of a non-negative integer
    bl = int(x).bit_length()
    return (bl + 6) // 7  # ceil(bl / 7)

def _flatten_ints(obj) -> np.ndarray:
    # robustly flatten ints from numpy arrays OR lists/tuples/(value,count) pairs
    if isinstance(obj, np.ndarray):
        return obj.astype(np.int64, copy=False).reshape(-1)
    flat = []
    if isinstance(obj, (list, tuple)):
        for el in obj:
            if isinstance(el, (list, tuple, np.ndarray)):
                flat.extend(np.asarray(el, dtype=np.int64).reshape(-1).tolist())
            else:
                flat.append(int(el))
    else:
        flat = [int(obj)]
    return np.asarray(flat, dtype=np.int64)

def _varint_stream_cost(u64: np.ndarray, rle_hint: bool | None) -> int:
    """
    Cost proxy in *bytes* using varint lengths.
    - rle_hint=True: force RLE
    - rle_hint=False: force raw
    - rle_hint=None: choose the cheaper of the two (AUTO)
    """
    u64 = np.asarray(u64, dtype=np.uint64).reshape(-1)
    if u64.size == 0:
        return 0

    # raw
    raw_cost = int(np.sum([_varint_len_u64(int(v)) for v in u64.tolist()]))

    # rle
    pairs = _flatten_ints(rle_encode(u64.astype(np.int64, copy=False)))
    # pairs is a flattened [val,count,val,count,...] sequence (or becomes that via _flatten_ints)
    rle_cost = int(np.sum([_varint_len_u64(int(v)) for v in pairs.tolist()]))

    if rle_hint is True:
        return rle_cost
    if rle_hint is False:
        return raw_cost
    # AUTO → pick cheaper
    return min(raw_cost, rle_cost)

def _choose_control_mode() -> str:
    """
    Returns 'AUTO' or 'MANUAL'.
    Full Auto = pick all defaults/auto heuristics and run end-to-end (including compression).
    Manual   = current interactive flow.
    """
    print("\n🎛️ Control")
    print("  [A] Full Auto — one-shot encode with smart defaults")
    print("  [M] Manual Control — step through options")
    sel = (input("Select [A/M]: ").strip().upper() or "A")
    return "AUTO" if sel == "A" else "MANUAL"

def _effective_tags_from_payload(encoded: dict, scheme_internal: str) -> tuple[str, str]:
    """
    Returns (lp_tag, rle_tag) from the *encoded* payload so names/logs reflect
    what actually happened (AUTO resolved).
      - lp_tag: 'lp0' if OFF, else 'lpN'
      - rle_tag: 'rle0' or 'rle1'
    Works for both standard ('time_intzz_v1') and base-i ('basei_sparse_v1').
    """
    # RLE
    rle_bool = bool(encoded.get("rle", False))
    rle_tag = f"rle{1 if rle_bool else 0}"

    # LP (standard) vs RACLP (base-i)
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

def _print_final_sizes(input_wav_path: str, alak_path: str, compressed_path: str | None):
    # Resolve absolute paths and compute a shared base dir for pretty printing
    paths = [os.path.abspath(input_wav_path), os.path.abspath(alak_path)]
    if compressed_path:
        paths.append(os.path.abspath(compressed_path))
    try:
        base_dir = os.path.commonpath(paths)
    except Exception:
        base_dir = os.path.dirname(os.path.abspath(alak_path)) or "."

    # Sizes
    wav_size  = os.path.getsize(input_wav_path)
    alak_size = os.path.getsize(alak_path)
    comp_size = os.path.getsize(compressed_path) if compressed_path else None

    # Short (relative) names
    wav_name  = os.path.relpath(input_wav_path,  base_dir)
    alak_name = os.path.relpath(alak_path,       base_dir)
    comp_name = os.path.relpath(compressed_path, base_dir) if compressed_path else None

    print(f"\n📁 Directory: {base_dir}")
    print(f"🎧 Source WAV: {wav_name}  ({_human(wav_size)})")
    print(f"📂 Original  : {alak_name}  ({_human(alak_size)})")

    if comp_size is not None:
        print(f"📦 Compressed: {comp_name}  ({_human(comp_size)})")

        # Ratio 1: .alak → compressed
        red1   = (1 - comp_size / alak_size) * 100 if alak_size else 0.0
        ratio1 = (alak_size / comp_size) if comp_size else float("inf")
        frac1  = (comp_size / alak_size) if alak_size else 0.0
        print(f"\n📉 Reduced by (alak→cmp): {red1:.1f}%")
        print(f"📊 Ratio     (alak→cmp): {ratio1:.1f}x smaller")
        print(f"📏 Fraction  (alak→cmp): {frac1:.2f}")

        # Ratio 2: .wav → compressed (end-to-end)
        red2   = (1 - comp_size / wav_size) * 100 if wav_size else 0.0
        ratio2 = (wav_size / comp_size) if comp_size else float("inf")
        frac2  = (comp_size / wav_size) if wav_size else 0.0
        print(f"\n🎯 End-to-End Reduction (wav→cmp): {red2:.1f}%")
        print(f"🎯 End-to-End Ratio     (wav→cmp): {ratio2:.1f}x smaller")
        print(f"🎯 End-to-End Fraction  (wav→cmp): {frac2:.2f}")

    else:
        # No compression: show wav → .alak (useful when user skips compression)
        red   = (1 - alak_size / wav_size) * 100 if wav_size else 0.0
        ratio = (wav_size / alak_size) if alak_size else float("inf")
        frac  = (alak_size / wav_size) if wav_size else 0.0
        print(f"\n📉 Reduction (wav→alak): {red:.1f}%")
        print(f"📊 Ratio     (wav→alak): {ratio:.1f}x smaller")
        print(f"📏 Fraction  (wav→alak): {frac:.2f}")
   


# ---------- Base-i RACLP AUTO (returns tuple: (use?, chosen_order)) ----------
def _decide_raclp_auto(signal: np.ndarray, level: int, rle_mode_hint: bool | None = None, sr: int | None = None, dp_level: str | None = None) -> tuple[bool, int]:
    """
    Size-aware RACLP Auto for Base-i.
    Tries orders in {2,4,8,12,16} if enough data.
    Score = varint-byte cost of FOUR packed streams (idx/val real/imag after residuals) + coeff penalty.
    """
    x = np.asarray(signal, dtype=np.float32)
    n = x.size
    if n <= 80:
        return False, 0

    L = max(1, int(level))
    idx = np.arange(n, dtype=np.int64)
    rclass = (idx // L) % 4
    cycle = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex64)
    y = x.astype(np.complex64) * cycle[rclass]

    raw_candidates = [2, 4, 8, 12, 16]
    cands = [p for p in raw_candidates if n >= (20 * p + 16)]
    if not cands:
        cands = [2, 4, 8] if n >= 160 else [2, 4]

    alpha = 16.0
    best_score = float("inf")
    best_p = 0
    scores = []

    for p in cands:
        # LS in rotated domain
        try:
            X = np.stack([y[p - j - 1 : n - j - 1] for j in range(p)], axis=1)
        except Exception:
            continue
        y_tgt = y[p:]
        XtX = X.conj().T @ X
        lam = (1e-6 * float(np.trace(XtX).real) / max(p, 1))
        try:
            coeffs = np.linalg.solve(XtX + lam * np.eye(p, dtype=np.complex64),
                                     X.conj().T @ y_tgt)
        except Exception:
            continue

        pred = np.zeros(n, dtype=np.complex64)
        pred[p:] = (X @ coeffs).astype(np.complex64, copy=False)
        res = y - pred

        # quantize residuals then sparse-pack (indices & values)
        q_r = np.rint(res.real).astype(np.int32, copy=False)
        q_i = np.rint(res.imag).astype(np.int32, copy=False)

        ir = np.nonzero(q_r)[0].astype(np.int64); vr = q_r[ir].astype(np.int64)
        ii = np.nonzero(q_i)[0].astype(np.int64); vi = q_i[ii].astype(np.int64)

        # stream cost as varint bytes after Δ→zz (+ RLE if it helps / as hinted)
        def _packed_cost(arr: np.ndarray) -> int:
            if arr.size == 0: 
                return 0
            u64 = zigzag(deltas(arr.astype(np.int64))).astype(np.uint64)
            return _varint_stream_cost(u64, rle_mode_hint)

        stream_cost = _packed_cost(ir) + _packed_cost(ii) + _packed_cost(vr) + _packed_cost(vi)
        score = stream_cost + alpha * p
        scores.append((p, score))
        if score < best_score:
            best_score = score
            best_p = p

    if best_p == 0:
        return False, 0

    # Baseline (p=0): pack raw y (indices/values for real & imag) without RACLP
    q_r0 = np.rint(y.real).astype(np.int32, copy=False)
    q_i0 = np.rint(y.imag).astype(np.int32, copy=False)
    ir0 = np.nonzero(q_r0)[0].astype(np.int64); vr0 = q_r0[ir0].astype(np.int64)
    ii0 = np.nonzero(q_i0)[0].astype(np.int64); vi0 = q_i0[ii0].astype(np.int64)

    def _base_cost(arrs):
        total = 0
        for arr in arrs:
            u64 = zigzag(deltas(arr.astype(np.int64))).astype(np.uint64)
            total += _varint_stream_cost(u64, rle_mode_hint)
        return total

    base_cost = _base_cost([ir0, ii0, vr0, vi0])

        # Turn on only if we actually improve
    if best_score >= 0.95 * base_cost:
        return False, 0

    # Build a near-tie set
    slack = best_score * 1.02  # 2% band
    near = sorted([p for p, sc in scores if sc <= slack])

    # Probe (try at most 4 orders: largest three + best)
    cand = sorted(set(near + [best_p]))[-4:]
    probed = _probe_pick_order(
        signal=signal,
        sr=sr if sr is not None else 48000,   # safe fallback
        dp_level=dp_level,                     # pass through actual dp you use in encode
        scheme_internal="basei_sparse_v1",
        rle_mode=rle_mode_hint,
        level=level,
        orders=cand,
        basei_mode=True,
    )
    if probed is not None:
        # optional log:
        # print(f"🧪 Probe pick (RACLP) → order {probed} among {cand}")
        return True, int(probed)

    # Fallback: smallest order within a tight tie
    tight_slack = best_score * 1.005
    chosen = min(p for p, sc in scores if sc <= tight_slack)
    return True, int(chosen)


# ---------- Standard LP AUTO (returns tuple: (use?, chosen_order)) ----------
def _decide_lp_auto_standard(signal: np.ndarray, dp_level: str, rle_mode_hint: bool | None = None, sr: int | None = None) -> tuple[bool, int]:
    """
    Size-aware LP Auto for standard pipeline.
    Tries orders in {2,4,8,12,16,24} if enough data.
    Score = varint-byte cost after residuals → Δ → zz (+ RLE if hinted/beneficial) + small coeff penalty.
    Returns (use_lp, chosen_order).
    """
    if dp_level in (None, "raw"):
        return False, 0

    q, _scale = quantize_to_ints(signal, dp_level)
    q = q.astype(np.float64, copy=False)
    n = int(q.size)
    if n <= 80:
        return False, 0

    raw_candidates = [2, 4, 8, 12, 16, 24]
    cands = [p for p in raw_candidates if n >= (20 * p + 16)]
    if not cands:
        cands = [2, 4, 8] if n >= 160 else [2, 4]

    best_score = float("inf")
    best_p = 0
    scores = []

    alpha = 2.0  # small bias against very large orders

    for p in cands:
        X = np.stack([q[p - j - 1 : n - j - 1] for j in range(p)], axis=1)
        y_tgt = q[p:]

        XtX = X.T @ X
        lam = (1e-6 * float(np.trace(XtX)) / max(p, 1))
        try:
            coeffs = np.linalg.solve(XtX + lam * np.eye(p), X.T @ y_tgt)
        except Exception:
            continue

        pred = np.zeros(n, dtype=np.float64)
        pred[p:] = (X @ coeffs)

        # residuals in quantized-int domain
        r = q - pred
        r_int = np.rint(r).astype(np.int32, copy=False)

        # cost as varint bytes after Δ→zz (+ RLE if it helps / as hinted)
        u64 = zigzag(deltas(r_int).astype(np.int64)).astype(np.uint64)
        stream_cost = _varint_stream_cost(u64, rle_mode_hint)

        score = stream_cost + alpha * p
        scores.append((p, score))
        if score < best_score:
            best_score = score
            best_p = p

    if best_p == 0:
        return False, 0

    # Baseline (p=0): pack q directly
    u64_base = zigzag(deltas(q.astype(np.int32, copy=False)).astype(np.int64)).astype(np.uint64)
    base_cost = _varint_stream_cost(u64_base, rle_mode_hint)

        # Optional debug (now safe)
    try:
        delta_pct = (base_cost - best_score) / base_cost * 100.0 if base_cost > 0 else 0.0
        print(f"🔎 LP auto score: base={int(base_cost)} best(p={best_p})={int(best_score)} "
              f"Δ={delta_pct:.1f}% (rle_hint={'A' if rle_mode_hint is None else int(rle_mode_hint)})")
    except Exception:
        pass

    # Enable if improvement ≥2%, or near-tie (+0.5%)
    enable = (best_score < 0.98 * base_cost) or (best_score <= base_cost * 1.005)
    if not enable:
        return False, 0

    # Build a near-tie set around the winner (favor higher p in the set)
    slack = best_score * 1.02  # 2% band
    near = sorted([p for p, sc in scores if sc <= slack])
    # Prefer to test at most 4 orders: include the top 3 largest + the exact best
    cand = sorted(set(near + [best_p]))[-4:]

    # Tiny probe: actually encode + bz2 and pick the smallest
    probed = _probe_pick_order(
        signal=signal,
        sr=sr,                      # <-- add 'sr' to function args or close over it
        dp_level=dp_level,
        scheme_internal="time_intzz_v1",
        rle_mode=rle_mode_hint,
        level=1,                    # not used for standard
        orders=cand,
        basei_mode=False,
    )
    if probed is not None:
        return True, int(probed)

    # Fallback (no probe success): pick largest p within slack
    chosen = max(near)
    return True, int(chosen)

def _choose_rle_mode():
    print("\n🧱 RLE (Run-Length Encoding)")
    print("  [A] AUTO – Let encoder decide")
    print("  [0] OFF  – Disable run-length encoding")
    print("  [1] ON   – Force run-length encoding")
    sel = (input("Choose RLE mode [A/0/1]: ").strip().upper() or "A")
    return {"A": None, "0": False, "1": True}[sel]

def encode_file(input_path: str | None = None,
                output_path: str | None = None,
                dp_level: str | None = None):

    # --- Input WAV ---
    if not input_path:
        input_path = input("📂 Enter input .wav file path: ").strip()
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    # --- Load audio first (so dp_auto and AUTO heuristics can use it) ---
    print(f"\n🔍 Loading audio: {input_path}")
    signal, sr = sf.read(input_path, always_2d=False)
    if getattr(signal, "ndim", 1) > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)

    # --- NEW: Control mode ---
    control = _choose_control_mode()  # 'AUTO' or 'MANUAL'

    # Defaults (will be set either by AUTO or prompts)
    scheme_internal = "time_intzz_v1"
    level = 1
    use_raclp = False
    raclp_order = 0
    use_lp = False
    lp_order = 0
    rle_mode = None  # None = AUTO

    if control == "AUTO":
        # Precision: dp_auto if available, else dp4
        if dp_level in (None, ""):
            if HAS_DPAUTO:
                picked = dp_auto(signal, sample_rate=sr)  # -> 1..4
                dp_level = f"dp{int(picked)}"
                print(f"🤖 dp_auto selected → {dp_level}")
            else:
                dp_level = "dp4"
                print("⚠️ dp_auto unavailable; using dp4.")
        is_raw = (dp_level in (None, "raw"))

        # Pipeline AUTO (same lightweight heuristic you already use)
        if not is_raw:
            from numpy.fft import rfft
            head = signal[:min(len(signal), sr)]
            spec = np.abs(rfft(head)) + 1e-9
            use_basei = (np.var(np.log(spec)) < 2.0)
            scheme_internal = "basei_sparse_v1" if use_basei else "time_intzz_v1"
        else:
            scheme_internal = "time_intzz_v1"

        # Predictors AUTO
        if scheme_internal == "basei_sparse_v1":
            level = 1  # sensible default
            use_raclp, raclp_order = _decide_raclp_auto(signal, level=level, rle_mode_hint=rle_mode, sr=sr, dp_level=dp_level)
            if use_raclp:
                print(f"🤖 RACLP AUTO → on (order={raclp_order})")
            else:
                print("🤖 RACLP AUTO → off")
        else:
            use_lp, lp_order = _decide_lp_auto_standard(signal, dp_level=dp_level, rle_mode_hint=rle_mode, sr=sr)
            if use_lp:
                print(f"🤖 LP AUTO (standard) → on (order={lp_order})")
            else:
                print("🤖 LP AUTO (standard) → off")

        # RLE AUTO (None already means AUTO)
        rle_mode = None

        # --- Print Effective Settings ---
        print(f"\n🧱 RLE mode: AUTO\n")

    else:
        # ----- MANUAL CONTROL (current interactive flow) -----
        # --- Precision (with dp_auto option) ---
        if not dp_level:
            print("\n🧠 Choose compression precision:")
            print(" [A] dp_auto")
            print(" [0] raw  [4] dp4  [3] dp3  [2] dp2  [1] dp1")
            sel = (input("Enter selection [A/0–4]: ").strip() or "A").upper()
            if sel == "A":
                if HAS_DPAUTO:
                    picked = dp_auto(signal, sample_rate=sr)  # -> 1..4
                    dp_level = f"dp{int(picked)}"
                    print(f"🤖 dp_auto selected → {dp_level}")
                else:
                    print("⚠️ dp_auto unavailable; falling back to dp4.")
                    dp_level = "dp4"
            else:
                dp_level = {"0": "raw", "4": "dp4", "3": "dp3", "2": "dp2", "1": "dp1"}.get(sel, "dp4")

        is_raw = (dp_level in (None, "raw"))

        # --- Pipeline & RLE menus ---
        pipeline_label = _choose_pipeline()
        scheme_choice = PIPELINE_MAP[pipeline_label]  # "auto" | "time_intzz_v1" | "basei_sparse_v1"
        rle_mode = _choose_rle_mode()
        mode_label = {None: "AUTO", False: "OFF", True: "ON"}[rle_mode]
        print(f"🧱 RLE mode: {mode_label}\n")

        # Choose pipeline for real
        if not is_raw:
            if scheme_choice == "auto":
                from numpy.fft import rfft
                head = signal[:min(len(signal), sr)]
                spec = np.abs(rfft(head)) + 1e-9
                use_basei = (np.var(np.log(spec)) < 2.0)
                scheme_internal = "basei_sparse_v1" if use_basei else "time_intzz_v1"
            else:
                scheme_internal = scheme_choice

            # Path-specific prompts
            if scheme_internal == "basei_sparse_v1":
                level_input = input("Enter rotation level for Base-i [1]: ").strip() or "1"
                try:
                    level = max(1, int(level_input))
                except ValueError:
                    level = 1

                mode = (input("🧮 RACLP (Base-i)? [A]uto / [Y]es / [n]o: ").strip().lower() or "a")
                if mode == "a":
                    use_raclp, raclp_order = _decide_raclp_auto(signal, level=level, rle_mode_hint=rle_mode, sr=sr, dp_level=dp_level)
                    print(f"🤖 RACLP AUTO → {'on' if use_raclp else 'off'}"
                          f"{f' (order={raclp_order})' if use_raclp else ''}")
                elif mode in ("y", "yes"):
                    raw = input("LP order for Base-i (complex) [4]: ").strip() or "4"
                    try:
                        raclp_order = max(1, int(raw))
                    except ValueError:
                        raclp_order = 4
                    use_raclp = True
                else:
                    use_raclp = False
                    raclp_order = 0
            else:
                mode = (input("🧮 Linear Prediction (standard)? [A]uto / [Y]es / [n]o: ").strip().lower() or "a")
                if mode == "a":
                    use_lp, lp_order = _decide_lp_auto_standard(signal, dp_level=dp_level, rle_mode_hint=rle_mode, sr=sr)
                    print(f"🤖 LP AUTO (standard) → {'on' if use_lp else 'off'}"
                          f"{f' (order={lp_order})' if use_lp else ''}")
                elif mode in ("y", "yes"):
                    raw = input("LP order for standard [4]: ").strip() or "4"
                    try:
                        lp_order = max(1, int(raw))
                    except ValueError:
                        lp_order = 4
                    use_lp = True
                else:
                    use_lp = False
                    lp_order = 0

    # ---- Plan line in filename order (same as you already do) ----
    dp_tag = dp_level or "raw"
    pipe_tag = "basei" if scheme_internal == "basei_sparse_v1" else "std"
    level_tag = (f".L{level}" if scheme_internal == "basei_sparse_v1" else "")
    if scheme_internal == "basei_sparse_v1":
        pred_tag = f"lp{raclp_order}" if use_raclp else "lp0"
    else:
        pred_tag = f"lp{lp_order}" if use_lp else "lp0"
    rle_plan_tag = "A" if rle_mode is None else ("1" if rle_mode else "0")
    print(f"\n⚙️ Encoding — {dp_tag}.{pipe_tag}{level_tag}.{pred_tag}.rle{rle_plan_tag}\n")

    # --- Encode ---
    encoded = encode_basei_sparse(
        signal,
        dp=None if dp_level == "raw" else dp_level,
        scheme=scheme_internal,
        use_rle=rle_mode,             # None= AUTO, False=OFF, True=ON
        prune_counts=0,
        level=level,
        use_raclp=use_raclp,          # Base-i predictor
        use_lp=use_lp,                # Standard predictor
        lp_order=(raclp_order if scheme_internal == "basei_sparse_v1" else lp_order),
    )

    # Effective line (post-encode): derive from payload so AUTO is resolved
    lp_tag, rle_tag = _effective_tags_from_payload(encoded, scheme_internal)
    eff_line = f"✅ Effective — {dp_tag}.{pipe_tag}{level_tag}.{lp_tag}.{rle_tag}"
    print(eff_line + "\n")

    # --- Build auto filename from EFFECTIVE tags ---
    base = os.path.splitext(os.path.basename(input_path))[0]
    name_tags = [dp_tag, pipe_tag]
    if pipe_tag == "basei":
        name_tags.append(f"L{level}")
    name_tags.append(lp_tag)      # e.g., lp0 / lp2 / lp4 …
    name_tags.append(rle_tag)     # rle0 / rle1
    auto_name = f"{base}." + ".".join(name_tags) + ".alak"

    # --- Decide final output name (honor provided output_path; skip prompt in Full Auto) ---
    if output_path:
        final_out = output_path
    else:
        if control == "AUTO":
            final_out = auto_name
            print(f"💾 Auto filename: {final_out}")
        else:
            resp = input(f"\n💾 Use auto filename '{auto_name}'? [Y/n]: ").strip().lower()
            final_out = auto_name if resp in ("", "y", "yes") else (input("Enter output .alak filename: ").strip() or auto_name)

    # --- Save container (.alak) ---
    print(f"💾 Saving .alak → {final_out}")
    save_alak_file(
        path=final_out,
        sample_rate=int(sr),
        original_length=len(signal),
        dp=dp_level,
        compressed_data=encoded,
    )
    print(f"✅ .alak saved: {final_out}")

    # --- Compression ---
    compressed_path = None
    if control == "AUTO":
        # No prompts; bz2 default, quiet
        try:
            compressed_path = _compress_quiet(final_out, "bz2")
        except Exception as e:
            print(f"❌ Compression failed: {e}")
        _print_final_sizes(input_path, final_out, compressed_path)
    else:
        do_comp = input("\n📦 Compress now? [Y/n]: ").strip().lower()
        if do_comp in ("", "y", "yes"):
            print("Choose compression method:")
            print(" [1] bz2 (default)\n [2] gzip\n [3] lzma / xz\n [4] zstd")
            msel = input("Enter method number [1–4]: ").strip()
            method = {"1": "bz2", "2": "gzip", "3": "lzma", "4": "zstd"}.get(msel, "bz2")
            try:
                compressed_path = compress_alak_path(final_out, method)
            except Exception as e:
                print(f"❌ Compression failed: {e}")
            _print_final_sizes(input_path, final_out, compressed_path)
        else:
            print("ℹ️ Skipped compression.")
            _print_final_sizes(input_path, final_out, None)

    # --- Metrics at the very end ---
    recon = decode_basei_sparse(encoded)[:len(signal)]

    # --- Band-limited SNR report ---
    try:
        from evaluation.band_snr import make_band_snr_report, format_report_text, DEFAULT_BANDS_BROAD5, SPEECH_FOCUS, OCTAVE_LIKE
        # Pick a preset (or pass your own list of bands)
        rep1 = make_band_snr_report(signal, recon, sr, bands=DEFAULT_BANDS_BROAD5, align=True)
        print(format_report_text(rep1, title="Band SNR – Broad (5 bands)"))

        rep2 = make_band_snr_report(signal, recon, sr, bands=SPEECH_FOCUS, align=True)
        print(format_report_text(rep2, title="Band SNR – Speech Focus"))

        # Optional third set
        # rep3 = make_band_snr_report(signal, recon, sr, bands=OCTAVE_LIKE, align=True)
        # print(format_report_text(rep3, title="Band SNR – Octave-ish"))
    except Exception as e:
        print(f"(Band-limited SNR report skipped: {e})")

    # Classic point-by-point metrics (no gain/delay)
    mse = float(np.mean((signal - recon) ** 2))
    mae = float(np.mean(np.abs(signal - recon)))
    snr_classic = 10.0 * math.log10((np.mean(signal ** 2) + 1e-12) / (mse + 1e-12))

    print("\n📊 Compression Quality Metrics")
    print("================================")
    print(f"📏 MSE: {mse:.8f}")
    print(f"🎯 MAE: {mae:.6f}")

    # Robust SNR (gain/delay alignment) + Base-i diagnostics when applicable
    try:
        from evaluation.metrics import evaluate_metrics_basei

        level_eff = int(encoded.get("level", level))  # fall back to the chosen level
        rep = evaluate_metrics_basei(signal, recon, sample_rate=int(sr), level=level_eff)

        print(f"📶 SNR (classic): {snr_classic:.2f} dB")
        print(f"🔧 SNR (robust) : {rep['robust_snr_db']:.2f} dB")
        print("================================")

        if scheme_internal == "basei_sparse_v1":
            pcs = rep.get("per_class_snr_db")
            if pcs is not None:
                print("\n🎛 Base-i diagnostics")
                print(f"  vector SNR (complex): {rep['vector_snr_db']:.2f} dB")
                print(
                    f"  per-class SNRs: c0={pcs[0]:.2f} dB  c1={pcs[1]:.2f} dB  "
                    f"c2={pcs[2]:.2f} dB  c3={pcs[3]:.2f} dB"
                )
                print(
                    f"  LSD: {rep['lsd_db']:.3f} dB   "
                    f"delay={rep['align_delay_samples']}   gain={rep['gain']:.6f}"
                )
    except Exception as e:
        # Still show classic SNR even if robust/Base-i path fails
        print(f"📶 SNR (classic): {snr_classic:.2f} dB")
        print("================================")
        print(f"(Robust/Base-i metrics skipped: {e})")
