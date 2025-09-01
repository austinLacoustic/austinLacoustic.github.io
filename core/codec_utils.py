# core/codec_utils.py
import numpy as np

# --- Varint (LEB128) encode/decode for nonnegative ints ---
def varint_encode(u64: np.ndarray) -> bytes:
    """
    Encode a nonnegative integer array (uint64/int64 >= 0) to LEB128 bytes.
    """
    arr = np.asarray(u64, dtype=np.uint64).ravel()
    out = bytearray()
    for x in arr:
        v = int(x)
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)  # continuation
            else:
                out.append(b)
                break
    return bytes(out)

def varint_decode(data: bytes) -> np.ndarray:
    """
    Decode LEB128 bytes back to a uint64 numpy array.
    """
    out = []
    val = 0
    shift = 0
    for byte in data:
        byte = int(byte)
        val |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            out.append(val)
            val = 0
            shift = 0
    return np.asarray(out, dtype=np.uint64)


# ---------- quantization ----------
def dp_to_scale(dp: str | int) -> int:
    # "dp3" -> 1000 ; 3 -> 1000
    if isinstance(dp, str) and dp.startswith("dp"):
        n = int(dp[2:])
    elif isinstance(dp, int):
        n = dp
    else:
        raise ValueError(f"Bad dp: {dp}")
    return 10 ** n

def quantize_to_ints(arr: np.ndarray, dp) -> tuple[np.ndarray, int]:
    scale = dp_to_scale(dp)
    q = np.rint(np.asarray(arr, dtype=np.float32) * scale).astype(np.int32)
    return q, scale

def quantize_to_ints_shaped(x: np.ndarray, dp: str, *, mode: str = "off",
                            beta: float = 0.85, seed: int | None = None):
    """
    mode: "off" | "tpdf" | "ns1" | "tpdf+ns1"
      - tpdf    : triangular PDF dither at ±1 LSB peak
      - ns1     : first-order error-feedback noise-shaping (stable for 0<beta<1)
      - tpdf+ns1: both (recommended for DP2)
    Returns (q_int:int32 array, scale:int)
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    x = x.astype(np.float32, copy=False)

    # same scale rule you use in quantize_to_ints
    if isinstance(dp, str) and dp.startswith("dp"):
        places = int(dp[2:] or "0")
        scale = 10 ** places
    else:
        raise ValueError(f"Unsupported dp '{dp}' for shaped quantizer")

    y = x.copy()

    # TPDF dither (adds ~±1 LSB peak triangular noise; doesn’t require decoder changes)
    if "tpdf" in mode:
        rng = np.random.default_rng(seed)
        # d ∈ [-1,1] scaled by 1 LSB
        d = (rng.random(y.size, dtype=np.float32) - rng.random(y.size, dtype=np.float32)) / float(scale)
        y = y + d

    # First-order noise shaping (error feedback)
    if "ns1" in mode:
        out = np.empty_like(y, dtype=np.int32)
        e_prev = 0.0
        s = float(scale)
        b = float(beta)
        # Simple tight loop; fast enough in Python for audio lengths
        for i in range(y.size):
            xi = y[i] + b * e_prev
            qi = int(np.rint(xi * s))
            out[i] = qi
            e_prev = xi - (qi / s)
        return out, scale

    # Fallback: plain rounding (no shaping)
    q = np.rint(y * scale).astype(np.int32, copy=False)
    return q, scale

def dequantize_from_ints(q: np.ndarray, dp_or_scale) -> np.ndarray:
    # Accepts "dpN" or an integer scale
    if isinstance(dp_or_scale, str):
        scale = dp_to_scale(dp_or_scale)
    else:
        scale = int(dp_or_scale)
    return (np.asarray(q, dtype=np.int32).astype(np.float32)) / float(scale)


# ---------- delta / zigzag ----------
def deltas(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    if a.size == 0:
        return a
    d = np.empty_like(a, dtype=np.int64)
    d[0] = a[0]
    if a.size > 1:
        d[1:] = a[1:] - a[:-1]
    return d

def undeltas(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.int64)
    if d.size == 0:
        return d
    out = np.empty_like(d, dtype=np.int64)
    out[0] = d[0]
    if d.size > 1:
        out[1:] = out[0] + np.cumsum(d[1:], dtype=np.int64)
    return out

def zigzag(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.int64)
    u = (n << 1) ^ (n >> 63)  # signed -> unsigned mapping
    return u.astype(np.uint64)

def unzigzag(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint64)
    n = (u >> 1) ^ (-(u & 1))
    return n.astype(np.int64)


# ---------- optional RLE over integers ----------
def rle_encode(vals: np.ndarray) -> list[list[int]]:
    vals = np.asarray(vals)
    out = []
    if vals.size == 0:
        return out
    run_val = int(vals[0]); run_len = 1
    for v in vals[1:]:
        v = int(v)
        if v == run_val and run_len < 2**31 - 1:
            run_len += 1
        else:
            out.append([run_val, run_len])
            run_val = v; run_len = 1
    out.append([run_val, run_len])
    return out

def rle_decode(pairs: list[list[int]]) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.int64)
    out = []
    for v, c in pairs:
        out.extend([int(v)] * int(c))
    return np.asarray(out, dtype=np.int64)
