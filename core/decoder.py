# core/decoder.py

import numpy as np
from core.codec_utils import undeltas, unzigzag, rle_decode, varint_decode, dequantize_from_ints

print(f"[decoder] using {__file__}")

# --- helper: unpack RLE( zigzag( deltas(seq) ) ) back to original ints ---
def _unpack_delta_zz_rle(pairs) -> np.ndarray:
    """
    pairs: list[[value,count],...] over uint64 zigzagged deltas (stored as ints)
    returns: 1-D np.int64 sequence (original indices or values)
    """
    if not pairs:
        return np.asarray([], dtype=np.int64)
    u = np.asarray(rle_decode(pairs), dtype=np.uint64)  # zigzagged deltas
    s = unzigzag(u)  # -> int64 deltas
    return undeltas(s).astype(np.int64, copy=False)

# base-i inverse cycle (conjugate of encode cycle)
_BASEI_INV = np.array([1+0j, 0-1j, -1+0j, 0+1j], dtype=np.complex64)  # [1, -1j, -1, 1j]

# ---------------------------
# helpers
# ---------------------------
def _fix_length(y: np.ndarray, N: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if N <= 0:
        return np.zeros((0,), dtype=np.float32)
    if y.size >= N:
        return y[:N].astype(np.float32, copy=False)
    return np.pad(y.astype(np.float32, copy=False), (0, N - y.size), mode="constant")

def _de_emphasis(y: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y.copy()
    out = np.empty_like(y)
    out[0] = y[0]
    a = float(alpha)
    for i in range(1, y.size):
        out[i] = y[i] + a * out[i-1]
    return out


# ---------------------------
# time_intzz_v1
# ---------------------------
def _decode_time_intzz_v1(payload: dict) -> np.ndarray:
    N_decl = int(payload.get("original_length", 0))
    scale  = float(payload.get("scale", 1.0)) or 1.0
    dp     = payload.get("dp")
    diff_order = int(payload.get("diff_order", 1))

    if payload.get("storage") == "vint":
        import base64
        byts = payload.get("bin", b"")
        if not isinstance(byts, (bytes, bytearray)):
            byts = base64.b64decode(byts)
        u64 = varint_decode(byts)                     # uint64 of Δ^k (zigzagged)
    else:
        if payload.get("rle", False) and ("signal_zz_rle" in payload):
            u64 = np.asarray(rle_decode(payload["signal_zz_rle"]), dtype=np.uint64)
        elif "signal_zz" in payload:
            u64 = np.asarray(payload["signal_zz"], dtype=np.uint64)
        else:
            return np.zeros((max(0, N_decl),), dtype=np.float32)

    # Undo zigzag → int64 deltas of order k
    d = unzigzag(u64)

    # Rebuild the integer sequence from Δ^k
    if diff_order == 1:
        q = undeltas(d).astype(np.int32, copy=False)
    elif diff_order == 2:
        if d.size == 0:
            q = np.asarray([], dtype=np.int32)
        else:
            # d is Δ² on d1[1:], so rebuild d1 first:
            d1 = np.empty(d.size + 1, dtype=np.int64)
            d1[0] = 0
            d1[1:] = undeltas(d)      # cumulative sum
            q = undeltas(d1).astype(np.int32, copy=False)
    else:
        # future-proof: fallback to Δ¹
        q = undeltas(d).astype(np.int32, copy=False)

    # De-residualize if LP (unchanged)
    if payload.get("lp", False):
        p = int(payload.get("lp_order", 0) or 0)
        coeffs = np.asarray(payload.get("lp_coeffs", []), dtype=np.float64)
        if p > 0 and coeffs.size == p:
            q_rec = q.astype(np.int64, copy=False)
            for i in range(p, q_rec.size):
                pred = int(np.rint(np.dot(coeffs, q_rec[i - p:i][::-1])))
                q_rec[i] = q_rec[i] + pred
            q = q_rec.astype(np.int32, copy=False)

    # Dequantize exactly
    if dp is not None and dp != "raw":
        y = dequantize_from_ints(q, dp).astype(np.float32, copy=False)
    else:
        y = (q.astype(np.float32) / scale)
    N_eff = N_decl if N_decl > 0 else int(y.size)
    pre = payload.get("preemph")
    if isinstance(pre, dict) and pre.get("type") == "one_pole_preemphasis":
        alpha = float(pre.get("alpha", 0.97))
        y = _de_emphasis(y, alpha)
    return _fix_length(y, N_eff)


# ---------------------------
# basei_sparse_v1 with RACLP
# ---------------------------
def _decode_basei_sparse_v1(payload: dict) -> np.ndarray:
    """
    Supports:
      • New four-field packed form (…_zz_rle)
      • Legacy indices/values form
    RACLP reconstruction when payload['raclp'] is True.
    """
    N = int(payload.get("original_length", 0))
    if N <= 0:
        return np.zeros((0,), dtype=np.float32)
    scale = float(payload.get("scale", 1.0)) or 1.0
    L = max(1, int(payload.get("level", 1)))
    # New: Check for RACLP
    use_raclp = payload.get("raclp", False)
    lp_order = payload.get("lp_order", 0)
    lp_coeffs = None
    if use_raclp and lp_order > 0:
        coeffs_real = np.asarray(payload["lp_coeffs_real"], dtype=np.float64)
        coeffs_imag = np.asarray(payload["lp_coeffs_imag"], dtype=np.float64)
        lp_coeffs = coeffs_real + 1j * coeffs_imag
    # --- NEW packed form?
    has_new = all(k in payload for k in (
        "real_idx_zz_rle","imag_idx_zz_rle","real_val_zz_rle","imag_val_zz_rle"
    ))
    if has_new:
        idx_r = _unpack_delta_zz_rle(payload["real_idx_zz_rle"])
        idx_i = _unpack_delta_zz_rle(payload["imag_idx_zz_rle"])
        val_r = _unpack_delta_zz_rle(payload["real_val_zz_rle"]).astype(np.int32, copy=False)
        val_i = _unpack_delta_zz_rle(payload["imag_val_zz_rle"]).astype(np.int32, copy=False)
    else:
        # --- LEGACY fallback ---
        # indices as plain lists; values maybe RLE'd (values only)
        idx_r = np.asarray(payload.get("real_indices", []), dtype=np.int64)
        idx_i = np.asarray(payload.get("imag_indices", []), dtype=np.int64)
        rv = payload.get("real_values", [])
        iv = payload.get("imag_values", [])
        if payload.get("rle", False):
            val_r = rle_decode(rv).astype(np.int32, copy=False)
            val_i = rle_decode(iv).astype(np.int32, copy=False)
        else:
            val_r = np.asarray(rv, dtype=np.int32)
            val_i = np.asarray(iv, dtype=np.int32)
    # scatter to quantized buffers (residuals if RACLP)
    q_r = np.zeros((N,), dtype=np.int32)
    q_i = np.zeros((N,), dtype=np.int32)
    if idx_r.size and val_r.size:
        m = min(idx_r.size, val_r.size)
        q_r[idx_r[:m]] = val_r[:m]
    if idx_i.size and val_i.size:
        m = min(idx_i.size, val_i.size)
        q_i[idx_i[:m]] = val_i[:m]
    # dequantize
    y_r = q_r.astype(np.float32) / scale
    y_i = q_i.astype(np.float32) / scale
    # Rebuild complex residuals (rotated domain)
    y_res = (y_r + 1j * y_i).astype(np.complex64, copy=False)
    # RACLP: iterative reconstruction from residuals
    if use_raclp and (lp_coeffs is not None) and (N > int(lp_order) > 0):
        p = int(lp_order)
        coeffs = lp_coeffs.astype(np.complex64, copy=False)

        # y_rot will hold the reconstructed *rotated-domain* signal
        y_rot = np.zeros(N, dtype=np.complex64)

        # Seed: encoder only predicts for i >= p, so the first p samples are pure residuals
        # (predicted=0 at the encoder), hence copy through.
        upto = min(p, N)
        y_rot[:upto] = y_res[:upto]

        # For i >= p: pred = dot(coeffs, [y_rot[i-1], ..., y_rot[i-p]])
        # (use same lag ordering as the encoder)
        for i in range(p, N):
            v = y_rot[i - p : i][::-1]   # [y_{i-1}, y_{i-2}, ..., y_{i-p}]
            pred = np.dot(coeffs, v)
            y_rot[i] = y_res[i] + pred

        y_complex = y_rot  # final rotated-domain reconstruction
    else:
        # No RACLP; residuals are the rotated-domain signal
        y_complex = y_res
    # Now extract real/imag from reconstructed rotated y_complex
    y_r = np.real(y_complex)
    y_i = np.imag(y_complex)
    # inverse rotation (vectorized, robust)
    idx = np.arange(N, dtype=np.int64)
    rclass = (idx // L) % 4
    inv_cycle = np.array([1+0j, 0-1j, -1+0j, 0+1j], dtype=np.complex64)  # conj of encoder cycle
    out = np.real(y_complex * inv_cycle[rclass]).astype(np.float32, copy=False)
    return out

# ---------------------------
# time_raw_v1
# ---------------------------
def _decode_time_raw_v1(payload: dict) -> np.ndarray:
    N_decl = int(payload.get("original_length", 0))
    sig = payload.get("signal", [])
    if not sig:
        return np.zeros((max(0, N_decl),), dtype=np.float32)
    arr = np.asarray(sig, dtype=np.float32)
    return _fix_length(arr, N_decl if N_decl > 0 else arr.size)

# ---------------------------
# entry point
# ---------------------------
def decode_basei_sparse(payload: dict) -> np.ndarray:
    scheme = payload.get("scheme", "time_intzz_v1")
    if scheme == "time_intzz_v1":
        return _decode_time_intzz_v1(payload)
    if scheme == "basei_sparse_v1":
        return _decode_basei_sparse_v1(payload)
    if scheme == "time_raw_v1":
        return _decode_time_raw_v1(payload)
    # unknown: try scalar as a safe fallback
    try:
        return _decode_time_intzz_v1(payload)
    except Exception:
        N = int(payload.get("original_length", 0))
        return np.zeros((max(0, N),), dtype=np.float32)
    