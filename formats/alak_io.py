# formats/alak_io.py
from __future__ import annotations
import json, os
from typing import Any, Dict

try:
    import msgpack  # pip install "msgpack>=1.0,<2"
    HAS_MSGPACK = True
except Exception:
    HAS_MSGPACK = False
    msgpack = None  # type: ignore


def _build_container(sample_rate: int,
                     original_length: int,
                     dp: str | None,
                     compressed_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": "alak",
        "version": "beta1",
        "sample_rate": int(sample_rate),
        "original_length": int(original_length),
        "compressed_data": compressed_data,
        "dp": dp,
    }


def save_alak_file(*,
                   path: str,
                   sample_rate: int,
                   original_length: int,
                   dp: str | None,
                   compressed_data: Dict[str, Any],
                   binary: bool = False) -> None:
    """
    Save an ALAK container.
      - binary=False → compact JSON   (.alak)
      - binary=True  → msgpack bytes  (.alak.bin)
    The caller decides the filename extension; we write exactly what we're told.
    """
    container = _build_container(sample_rate, original_length, dp, compressed_data)

    if binary:
        if not HAS_MSGPACK:
            raise RuntimeError("MsgPack not available. Install with: pip install 'msgpack>=1.0,<2'")
        with open(path, "wb") as f:
            # use_bin_type=True => bytes keep bytes type, raw=False on load later
            f.write(msgpack.packb(container, use_bin_type=True))
        return

    # JSON text, compact (no spaces)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(container, f, ensure_ascii=False, separators=(",", ":"))


def load_alak_file(path: str) -> Dict[str, Any]:
    """
    Load either JSON (.alak) or MsgPack (.alak.bin).
    Heuristic: If first non-whitespace byte is '{' → JSON; else try msgpack.
    """
    # Peek a couple bytes
    with open(path, "rb") as fb:
        head = fb.read(2)
    # JSON if starts with '{' or whitespace then '{'
    if head[:1] in (b"{", b" ", b"\t", b"\n", b"\r"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # fallthrough: maybe it's msgpack despite '{' heuristic failing
            pass

    if not HAS_MSGPACK:
        raise RuntimeError("File is not JSON; msgpack required to load. Install: pip install 'msgpack>=1.0,<2'")

    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)
