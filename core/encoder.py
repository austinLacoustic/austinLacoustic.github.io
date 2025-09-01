# core/encoder.py

import numpy as np
import base64, os
import core.codec_utils as cu

print(f"[encoder] using {__file__}")

# ---------- RLE heuristics ----------
def _run_stats_uint64(arr: np.ndarray):
    """Return (n, runs, avg_run_len, zero_ratio, max_run)."""
    arr = np.asarray(arr)
    n = int(arr.size)
    if n == 0:
        return 0, 0, 0.0, 0.0, 0
    runs = 1
    zero_cnt = 1 if int(arr[0]) == 0 else 0
    cur = int(arr[0])
    cur_run = 1
    max_run = 1
    for x in arr[1:]:
        x = int(x)
        if x == cur:
            cur_run += 1
        else:
            runs += 1
            if cur_run > max_run:
                max_run = cur_run
            cur = x
            cur_run = 1
        if x == 0:
            zero_cnt += 1
    if cur_run > max_run:
        max_run = cur_run
    avg_run = n / runs
    zero_ratio = zero_cnt / n
    return n, runs, float(avg_run), float(zero_ratio), int(max_run)

def _should_use_rle(avg_run_len: float, zero_ratio: float, max_run: int) -> bool:
    """Heuristic tuned for JSON/MsgPack + bz2."""
    if avg_run_len >= 4.0: return True
    if zero_ratio >= 0.30: return True
    if max_run   >= 16:    return True
    return False

# --- helper: pack int sequence as deltas→zigzag→RLE (for Base-i streams) ---
def _pack_delta_zz_rle(arr: np.ndarray) -> list[list[int]]:
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        return []
    d = cu.deltas(arr)            # int64
    u = cu.zigzag(d)              # uint64
    return cu.rle_encode(u.astype(np.int64))  # store as ints in JSON

def _pre_emphasis(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size <= 1: 
        return x.copy()
    y = x.copy()
    y[1:] = y[1:] - float(alpha) * x[:-1]
    return y

# ---------- diff-order chooser (Δ¹ vs Δ²) ----------
def _varint_len_u64(x: int) -> int:
    """Unsigned LEB128 length (bytes) for a nonnegative int."""
    if x <= 0: return 1
    bl = int(x).bit_length()
    return (bl + 6) // 7

def _proxy_cost_varint(u64: np.ndarray) -> int:
    """Varint byte-count proxy for a uint64 stream."""
    if u64.size == 0: return 0
    return int(sum(_varint_len_u64(int(v)) for v in u64.tolist()))

def _proxy_cost_varint_or_rle(u64: np.ndarray, rle_hint: bool | None) -> int:
    """Choose raw varint vs RLE(varint) depending on hint (True/False/None=AUTO)."""
    raw = _proxy_cost_varint(u64)
    if rle_hint is False:
        return raw
    # RLE proxy
    pairs = cu.rle_encode(u64.astype(np.int64, copy=False))
    rle_bytes = 0
    for v, c in pairs:
        rle_bytes += _varint_len_u64(int(v))
        rle_bytes += _varint_len_u64(int(c))
    if rle_hint is True:
        return rle_bytes
    return min(raw, rle_bytes)

def _dzz_and_cost(q_i64: np.ndarray, use_rle_hint: bool | None, prefer_delta1_margin: float = 0.03):
    """
    Returns: (diff_order, chosen_u64, other_u64, scores_dict)
    Bias toward Δ¹; Δ² must win by >= prefer_delta1_margin.
    """
    # Δ¹
    d1  = cu.deltas(q_i64)
    zz1 = cu.zigzag(d1).astype(np.uint64, copy=False)

    # Δ² on d1[1:]
    if d1.size >= 2:
        d2_in = d1[1:]
        d2  = cu.deltas(d2_in)
        zz2 = cu.zigzag(d2).astype(np.uint64, copy=False)
    else:
        zz2 = np.asarray([], dtype=np.uint64)

    c1 = _proxy_cost_varint_or_rle(zz1, use_rle_hint)  # your proxy function
    c2 = _proxy_cost_varint_or_rle(zz2, use_rle_hint)

    if c2 < (1.0 - prefer_delta1_margin) * c1:
        order = 2; chosen = zz2; other = zz1
    else:
        order = 1; chosen = zz1; other = zz2

    scores = {"c1_bytes_proxy": int(c1), "c2_bytes_proxy": int(c2), "margin": float(prefer_delta1_margin)}
    return order, chosen, other, scores

# ---------------- Encoder ----------------
def encode_basei_sparse(signal: np.ndarray,
                        dp: str | None = "dp4",
                        scheme: str = "time_intzz_v1",
                        use_rle: bool | None = None,  # None = AUTO
                        prune_counts: int = 0,
                        *,
                        level: int = 1,
                        use_raclp: bool = True,  # For base-i
                        lp_order: int = 4,       # Shared for both pipelines
                        use_lp: bool = True):    # Enable LP for standard
    """
    Two pipelines (selected by `scheme`):
      • "time_intzz_v1" : time-domain quantize → prune → LP(optional)
                          → Δ^k → zigzag (+ RLE or vint)  [k auto in {1,2}]
      • "basei_sparse_v1" : rotate by base-i (level L), optional RACLP,
                            quantize real/imag, sparse-pack 4 streams with RLE.
    """
    # ---- normalize input ----
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal, dtype=np.float32)
    signal = signal.astype(np.float32, copy=False)
    orig_len = int(signal.shape[0])

    # --- optional HF pre-emphasis for scalar (time_intzz_v1) path ---
    preemph_cfg = None
    if scheme == "time_intzz_v1":
        # toggle via env; default OFF
        do_pre = os.environ.get("ALAK_PREEMPH", "0").lower() not in ("0","false","no")
        if do_pre:
            alpha = float(os.environ.get("ALAK_PREEMPH_ALPHA", "0.97"))
            signal = _pre_emphasis(signal, alpha)
            preemph_cfg = {"type": "one_pole_preemphasis", "alpha": alpha}

    # ---- RAW passthrough ----
    if dp is None or dp == "raw":
        return {
            "scheme": "time_raw_v1",
            "dp": "raw",
            "original_length": orig_len,
            "sample_layout": "mono_f32",
            "signal": signal.tolist(),
        }

    # ==========================================================
    # BASE-I SPARSE PIPELINE with optional RACLP
    # ==========================================================
    if scheme == "basei_sparse_v1" and (dp not in (None, "raw")):
        _, scale = cu.quantize_to_ints(signal, dp)
        L   = max(1, level)
        n   = orig_len
        idx = np.arange(n, dtype=np.int64)
        rclass = (idx // L) % 4

        y_complex = signal.astype(np.complex64)
        cycle = np.array([1+0j, 0+1j, -1+0j, 0-1j])
        y_complex *= cycle[rclass]

        lp_coeffs = None
        if use_raclp and n > lp_order:
            p = int(lp_order)
            X = np.stack([y_complex[p - j - 1 : n - j - 1] for j in range(p)], axis=1)
            y_tgt = y_complex[p:n]
            lp_coeffs = np.linalg.lstsq(X.astype(np.complex128),
                                        y_tgt.astype(np.complex128),
                                        rcond=None)[0]  # keep complex128
            predicted = np.zeros(n, dtype=np.complex64)
            for i in range(p, n):
                v = y_complex[i - p : i][::-1]
                predicted[i] = np.dot(lp_coeffs, v)
            y_complex[p:] = (y_complex[p:] - predicted[p:]).astype(np.complex64, copy=False)

        y_r = np.real(y_complex)
        y_i = np.imag(y_complex)

        q_r = np.rint(y_r * scale).astype(np.int32)
        q_i = np.rint(y_i * scale).astype(np.int32)

        th = int(prune_counts) if isinstance(prune_counts, (int, np.integer)) else 0
        if th > 0:
            if q_r.size: q_r[np.abs(q_r) < th] = 0
            if q_i.size: q_i[np.abs(q_i) < th] = 0

        idx_r = np.nonzero(q_r)[0].astype(np.int64); val_r = q_r[idx_r].astype(np.int64)
        idx_i = np.nonzero(q_i)[0].astype(np.int64); val_i = q_i[idx_i].astype(np.int64)

        real_idx_zz_rle = _pack_delta_zz_rle(idx_r)
        imag_idx_zz_rle = _pack_delta_zz_rle(idx_i)
        real_val_zz_rle = _pack_delta_zz_rle(val_r)
        imag_val_zz_rle = _pack_delta_zz_rle(val_i)

        return {
            "scheme": "basei_sparse_v1",
            "dp": dp,
            "scale": int(scale),
            "original_length": orig_len,
            "level": int(L),
            "sample_layout": "mono_qint_sparse",
            "rle": True,
            "raclp": use_raclp,
            "lp_order": lp_order if use_raclp else None,
            "lp_coeffs_real": np.real(lp_coeffs).astype(float).tolist() if use_raclp else None,
            "lp_coeffs_imag": np.imag(lp_coeffs).astype(float).tolist() if use_raclp else None,
            "real_idx_zz_rle": real_idx_zz_rle,
            "imag_idx_zz_rle": imag_idx_zz_rle,
            "real_val_zz_rle": real_val_zz_rle,
            "imag_val_zz_rle": imag_val_zz_rle,
        }

    # ==========================================================
    # STANDARD SCALAR PIPELINE (with optional LP)
    # ==========================================================
    # --- DP-aware quantization ---
    use_dp2_boost = (dp == "dp2") and bool(os.environ.get("ALAK_DP2_BOOST", "0") not in ("0","false","False"))

    try:
        from core.codec_utils import quantize_to_ints_shaped
        if use_dp2_boost:
            # best sounding combo: triangular dither + 1st-order shaping
            q, scale = quantize_to_ints_shaped(signal, dp, mode="tpdf+ns1", beta=0.85)
        else:
            from core.codec_utils import quantize_to_ints
            q, scale = quantize_to_ints(signal, dp)
    except Exception:
        # hard fallback if helper isn’t available
        from core.codec_utils import quantize_to_ints
        q, scale = quantize_to_ints(signal, dp)

    # Optional pruning in quantized domain
    th = int(prune_counts) if isinstance(prune_counts, (int, np.integer)) else 0
    if th > 0 and q.size:
        q[np.abs(q) < th] = 0

    # --- LP branch ---
    lp_coeffs = None
    if use_lp and len(q) > lp_order:
        p = int(lp_order)

        q_i64 = q.astype(np.int64, copy=False)
        rows = q_i64.size - p
        X = np.empty((rows, p), dtype=np.float64)
        for j in range(p):
            X[:, j] = q_i64[p - 1 - j : q_i64.size - 1 - j]
        y_tgt = q_i64[p:].astype(np.float64, copy=False)

        lam = 1e-8
        Xt = X.T
        A = Xt @ X + lam * np.eye(p)
        b = Xt @ y_tgt
        lp_coeffs = np.linalg.solve(A, b).astype(np.float64, copy=False)

        # Residualize with immutable context
        e = q_i64.copy()
        for i in range(p, q_i64.size):
            pred = int(np.rint(np.dot(lp_coeffs, q_i64[i - p : i][::-1])))
            e[i] = q_i64[i] - pred
        q = e.astype(np.int32, copy=False)

        # --- Δ¹/Δ² selection on residuals ---
        diff_order, dzz, _other, diff_scores = _dzz_and_cost(q.astype(np.int64, copy=False), use_rle)

        # --- RLE decision (AUTO/OFF/ON) ---
        rle_flag = bool(use_rle)
        rle_auto = False
        auto_reason = None
        if use_rle is None:
            _, _, avg_run, zero_ratio, max_run = _run_stats_uint64(dzz)
            rle_flag = _should_use_rle(avg_run, zero_ratio, max_run)
            rle_auto = True
            auto_reason = {
                "avg_run_len": round(float(avg_run), 3),
                "zero_ratio":  round(float(zero_ratio), 3),
                "max_run":     int(max_run),
                "decision":    "on" if rle_flag else "off",
            }

        payload = {
            "scheme": "time_intzz_v1",
            "diff_order": int(diff_order),
            "dp": dp,
            "scale": int(scale),
            "original_length": orig_len,
            "sample_layout": "mono_qint",
            "rle": bool(rle_flag),
            "lp": True,
            "lp_order": lp_order,
            "lp_coeffs": lp_coeffs.tolist(),
        }
        if rle_flag:
            payload["signal_zz_rle"] = cu.rle_encode(dzz.astype(np.int64, copy=False))
        else:
            payload["signal_zz"] = dzz.tolist()
        if rle_auto:
            payload["rle_auto"] = True
            payload["rle_auto_reason"] = auto_reason
        if preemph_cfg is not None:
            payload["preemph"] = preemph_cfg
        return payload
    
    # --- No-LP branch ---
    q_i64 = q.astype(np.int64, copy=False)
    diff_order, chosen_u64, _other_u64, diff_scores = _dzz_and_cost(q_i64, use_rle)
    dzz = chosen_u64  # <-- write the stream that matches diff_order
    diff_meta = {"diff_order": int(diff_order), "diff_order_scores": diff_scores}

    # Decide RLE (AUTO/OFF/ON) on the chosen stream
    rle_flag = bool(use_rle)
    rle_auto = False
    auto_reason = None
    if use_rle is None:
        _, _, avg_run, zero_ratio, max_run = _run_stats_uint64(dzz)
        rle_flag = _should_use_rle(avg_run, zero_ratio, max_run)
        rle_auto = True
        auto_reason = {
            "avg_run_len": round(float(avg_run), 3),
            "zero_ratio":  round(float(zero_ratio), 3),
            "max_run":     int(max_run),
            "decision":    "on" if rle_flag else "off",
        }

    # VINT fast-path only if final RLE is OFF
    if rle_flag is False:
        byts = cu.varint_encode(dzz)                     # bytes
        bin_b64 = base64.b64encode(byts).decode("ascii")
        payload = {
            "scheme": "time_intzz_v1",
            "storage": "vint",
            "bin": bin_b64,                           # JSON-safe
            "dp": dp,
            "scale": int(scale),
            "original_length": orig_len,
            "sample_layout": "mono_qint",
            "rle": False,
            "lp": False,
            "lp_order": None,
            "lp_coeffs": None,
            **diff_meta,                              # <— merge breadcrumbs here
        }
        if rle_auto:
            payload["rle_auto"] = True
            payload["rle_auto_reason"] = auto_reason
        if preemph_cfg is not None:
            payload["preemph"] = preemph_cfg
        return payload

    # Otherwise classic JSON (RLE ON)
    payload = {
        "scheme": "time_intzz_v1",
        "dp": dp,
        "scale": int(scale),
        "original_length": orig_len,
        "sample_layout": "mono_qint",
        "rle": True,
        "lp": False,
        "lp_order": None,
        "lp_coeffs": None,
        "signal_zz_rle": cu.rle_encode(dzz.astype(np.int64, copy=False)),
        **diff_meta,                                  # <— and here
    }
    if rle_auto:
        payload["rle_auto"] = True
        payload["rle_auto_reason"] = auto_reason
    if preemph_cfg is not None:
        payload["preemph"] = preemph_cfg
    return payload

