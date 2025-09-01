# formats/alak_loader.py
import os
import json
import gzip
import bz2
import lzma
import msgpack  # Ensure installed: pip install msgpack

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False

def _read_bytes(path: str) -> bytes:
    lower = path.lower()
    if lower.endswith(".alak") or lower.endswith(".alak.bin"):
        with open(path, "rb") as f:
            return f.read()
    if lower.endswith(".alak.gz") or lower.endswith(".alak.bin.gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    if lower.endswith(".alak.bz2") or lower.endswith(".alak.bin.bz2"):
        with bz2.open(path, "rb") as f:
            return f.read()
    if lower.endswith(".alak.xz") or lower.endswith(".alak.lzma") or \
       lower.endswith(".alak.bin.xz") or lower.endswith(".alak.bin.lzma"):
        with lzma.open(path, "rb") as f:
            return f.read()
    if lower.endswith(".alak.zst") or lower.endswith(".alak.bin.zst"):
        if not HAS_ZSTD:
            raise RuntimeError("zstd file but `zstandard` is not installed. pip install zstandard")
        d = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            return d.decompress(f.read())
    # Fallback: try plain
    with open(path, "rb") as f:
        return f.read()

def load_alak(path: str) -> dict:
    """Load .alak / .alak.bin / compressed variants and return parsed dict."""
    try:
        raw = _read_bytes(path)
    except Exception as e:
        raise ValueError(f"Failed to read file: {path} ({e})")
    
    # Try MessagePack first (binary .alak.bin)
    try:
        return msgpack.loads(raw, strict_map_key=False)  # strict_map_key=False for compatibility
    except (msgpack.UnpackException, ValueError):
        pass  # silently fall back to JSON

    # Fallback to JSON
    try:
        text = raw.decode("utf-8")
        return json.loads(text)
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")  # Handle BOM
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to decode as JSON or MessagePack: {path} ({e})")