# cli/compress_alak_file.py
import os, gzip, bz2, lzma, argparse

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False


def _human(n):
    units = ["B","KB","MB","GB","TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.2f} {u}"
        s /= 1024.0


def _compress_bytes(data: bytes, method: str) -> tuple[bytes, str]:
    m = method.lower()
    if m == "gzip":
        return gzip.compress(data), ".gz"
    if m == "bz2":
        return bz2.compress(data, compresslevel=9), ".bz2"
    if m in ("lzma", "xz"):
        return lzma.compress(data), ".xz"
    if m == "zstd":
        if not HAS_ZSTD:
            raise RuntimeError("zstd not available. pip install zstandard")
        return zstd.ZstdCompressor().compress(data), ".zst"
    raise ValueError(f"Unsupported compression method: {method}")


def compress_file(path: str, method: str = "bz2", verbose: bool = True) -> str:
    """
    Compress ANY .alak* file (JSON or MessagePack) as raw bytes.
    Produces '<original>.{gz|bz2|xz|zst}', preserving .alak/.alak.bin.
    Returns the output path.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        raw = f.read()

    comp, ext = _compress_bytes(raw, method)
    # Always append the compression extension; do NOT replace .alak/.bin
    out_path = path + ext

    with open(out_path, "wb") as f:
        f.write(comp)

    # sanity / report
    orig = os.path.getsize(path)
    comp_sz = os.path.getsize(out_path)
    frac = comp_sz / orig if orig else 0.0
    ratio = orig / comp_sz if comp_sz else float("inf")
    red = 100.0 * (1.0 - frac)

    if verbose:
        print(f"📂 Original:   {path}  ({_human(orig)})")
        print(f"📦 Compressed: {out_path}  ({_human(comp_sz)})")
        print(f"📉 Reduced by: {red:.1f}%")
        print(f"📊 Ratio:      {ratio:.1f}x smaller")
        print(f"📏 Fraction:   {frac:.2f}")

    if not os.path.exists(out_path) or comp_sz == 0:
        raise RuntimeError("Compression reported success but output file missing/empty.")

    return out_path


# Backwards-compatible alias (existing code may import this)
compress_alak_path = compress_file


# --- CLI entrypoint (argparse) ---
def main_cli():
    parser = argparse.ArgumentParser(
        description="Compress an .alak or .alak.bin file (raw bytes) to .gz/.bz2/.xz/.zst"
    )
    parser.add_argument("path", help="Path to .alak or .alak.bin")
    parser.add_argument(
        "-m", "--method",
        default="bz2",
        choices=["bz2", "gzip", "lzma", "xz", "zstd"],
        help="Compression method (default: bz2)",
    )
    args = parser.parse_args()

    try:
        out_path = compress_file(args.path, args.method, verbose=True)
        print(out_path)  # allow callers to capture the path
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
