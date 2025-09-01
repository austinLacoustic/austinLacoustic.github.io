# core/dp_auto.py
import numpy as np

def _quantize_time_domain(x: np.ndarray, dp: int) -> np.ndarray:
    if dp <= 0: return x.astype(np.float32, copy=False)
    scale = 10 ** dp
    q = np.rint(x * scale) / float(scale)
    return q.astype(np.float32, copy=False)

def _snr_db(ref: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    ref = ref.astype(np.float32, copy=False); test = test.astype(np.float32, copy=False)
    num = float(np.mean(ref**2) + eps); den = float(np.mean((ref - test)**2) + eps)
    import math; return 10.0 * math.log10(num/den)

def _is_mostly_silence(x: np.ndarray,
                       thresh: float = 3e-4,      # was 1e-3
                       min_ratio: float = 0.9):   # was 0.7
    if x.size == 0: return False
    return float(np.mean(np.abs(x) < thresh)) >= min_ratio

def dp_auto(signal: np.ndarray,
            sample_rate: int = 44100,
            target_snr_db: float = 60.0,
            floor_snr_db: float = 45.0,
            prefer_lower_dp: bool = True) -> int:
    if signal.ndim > 1: signal = signal[:, 0]
    x = signal.astype(np.float32, copy=False)

    # Only treat as silence if it's *very* quiet and *very* silent
    if _is_mostly_silence(x):
        # sanity check: if dp1 can't reach floor_snr, try the sweep
        y = _quantize_time_domain(x, 1)
        if _snr_db(x, y) >= floor_snr_db:
            return 1

    results = {}
    for dp in (1,2,3,4):
        y = _quantize_time_domain(x, dp)
        snr = _snr_db(x, y)
        mae = float(np.mean(np.abs(x - y)))
        results[dp] = (snr, mae)

    for dp in (1,2,3,4):
        if results[dp][0] >= target_snr_db:
            return dp

    best_dp = max(results.items(), key=lambda kv: kv[1][0])[0]
    best_snr = results[best_dp][0]
    if best_snr < floor_snr_db: return 4

    if prefer_lower_dp:
        ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
        top_snr = ranked[0][1][0]
        for dp, (snr, _) in ranked[::-1]:
            if top_snr - snr <= 1.0: return dp
    return best_dp
