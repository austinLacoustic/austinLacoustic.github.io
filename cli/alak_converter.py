# cli/alak_converter.py

import os
from formats.alak_io import save_alak_file
from formats.alak_loader import load_alak  # robust JSON/MsgPack/zip-aware loader

def _extract_fields(container: dict):
    """Return (sr, orig_len, dp, payload) from a loaded .alak container."""
    sr = int(container.get("sample_rate", 44100))
    orig_len = int(container.get("original_length", 0))
    dp = container.get("dp")
    payload = container.get("compressed_data", container)
    return sr, orig_len, dp, payload

def convert_json_to_binary(input_path=None, output_path=None):
    if input_path is None:
        input_path = input("📂 Enter JSON .alak path to convert to binary: ").strip()
    data = load_alak(input_path)  # handles JSON/MsgPack/and compressed variants
    sr, orig_len, dp, payload = _extract_fields(data)
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".alak.bin"
    save_alak_file(
        path=output_path,
        sample_rate=sr,
        original_length=orig_len,
        dp=dp,
        compressed_data=payload,
        binary=True,
    )
    print(f"✅ Converted to binary: {output_path}")

def convert_binary_to_json(input_path=None, output_path=None):
    if input_path is None:
        input_path = input("📂 Enter binary .alak.bin path to convert to JSON: ").strip()
    data = load_alak(input_path)
    sr, orig_len, dp, payload = _extract_fields(data)
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        # if input is "...alak.bin", strip only the last ".bin"
        if base.endswith(".alak"):
            output_path = base
        else:
            output_path = base + ".alak"
    save_alak_file(
        path=output_path,
        sample_rate=sr,
        original_length=orig_len,
        dp=dp,
        compressed_data=payload,
        binary=False,
    )
    print(f"✅ Converted to JSON: {output_path}")

def show_convert_menu():
    while True:
        print("\n🔄 Convert .alak Formats")
        print("=====================")
        print("[1] JSON to Binary (smaller size)")
        print("[2] Binary to JSON (editable)")
        print("[0] Back")
        choice = input("Select an option: ").strip()
        if choice == "1":
            convert_json_to_binary()
        elif choice == "2":
            convert_binary_to_json()
        elif choice == "0":
            return
        else:
            print("⚠️ Invalid selection.")
