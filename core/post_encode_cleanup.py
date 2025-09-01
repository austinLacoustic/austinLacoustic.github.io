# core/post_encode_cleanup.py
import numpy as np
import copy

def _to_fixed_str_list(arr, dp_digits: int):
    fmt = f"{{:.{dp_digits}f}}"
    # Convert to fixed strings so 0.123000047 -> "0.123"
    # We also strip any trailing zeros beyond dp if they exist
    out = []
    for x in np.asarray(arr):
        s = fmt.format(float(x))
        # Trim trailing zeros while keeping at most dp_digits places
        if '.' in s:
            s = s.rstrip('0').rstrip('.') if s.rstrip('0').rstrip('.') != '-0' else '0'
        out.append(s)
    return out

def cleanup_payload(payload: dict, dp: str | None) -> dict:
    """
    Non-destructive: returns a cleaned copy.
    Only trims textual bloat; values are preserved to chosen dp.
    """
    cleaned = copy.deepcopy(payload)
    if not dp or dp == "raw":
        # Nothing to do for raw
        return cleaned

    # Resolve dp digits
    try:
        dp_digits = int(dp[2:])  # "dp3" -> 3
    except Exception:
        dp_digits = 3

    cd = cleaned.get("compressed_data") or cleaned
    # If your encoder stores arrays under these keys:
    for key in ("real_even", "imag_odd"):
        if key in cd and isinstance(cd[key], list) and cd[key]:
            cd[key] = _to_fixed_str_list(cd[key], dp_digits)

    # Optional: if you store combined arrays elsewhere, add them here as needed.

    return cleaned
