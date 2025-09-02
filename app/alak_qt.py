# app/alak_qt.py
from __future__ import annotations
import os, sys, math, traceback, time, json, threading
from pathlib import Path
from PyQt6.QtCore import Qt

# --- repo root on sys.path (must come before core imports) ---
HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if not getattr(sys, "frozen", False):  # only in dev mode
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

# --- Qt ---
from PyQt6 import QtCore, QtGui, QtWidgets

# --- DSP/IO ---
import numpy as np
import soundfile as sf

# Optional audio backend used by the player dialog
try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    sd = None
    HAS_SD = False

# Optional msgpack (binary .alak.bin support)
try:
    import msgpack
    HAS_MSGPACK = True
except Exception:
    msgpack = None
    HAS_MSGPACK = False

# --- Core encode/decode/save/compress ---
from core.encoder import encode_basei_sparse
from core.decoder import decode_basei_sparse
from formats.alak_io import save_alak_file, load_alak_file
from cli.compress_alak_file import compress_alak_path

# Optional auto-helpers (import if present)
try:
    from core.dp_auto import dp_auto as _dp_auto
    dp_auto = _dp_auto
except Exception:
    dp_auto = None


# ---------- small helpers ----------
def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.2f} {u}"
        s /= 1024.0

def effective_tags_from_payload(encoded: dict, scheme_internal: str) -> tuple[str, str]:
    rle_bool = bool(encoded.get("rle", False))
    rle_tag = f"rle{1 if rle_bool else 0}"
    if scheme_internal == "basei_sparse_v1":
        on = bool(encoded.get("raclp", False))
        order = int(encoded.get("lp_order") or 0)
    else:
        on = bool(encoded.get("lp", False))
        order = int(encoded.get("lp_order") or 0)
    lp_tag = f"lp{(order if on and order > 0 else 0)}"
    return lp_tag, rle_tag

def choose_pipeline_auto(signal: np.ndarray, sr: int) -> str:
    """Tiny heuristic: Base-i for 'complex' content; Standard otherwise."""
    try:
        from numpy.fft import rfft
        head = signal[: min(len(signal), sr)]
        spec = np.abs(rfft(head)) + 1e-9
        use_basei = (np.var(np.log(spec)) < 2.0)
        return "basei_sparse_v1" if use_basei else "time_intzz_v1"
    except Exception:
        return "time_intzz_v1"

def _lp_auto_std_simple(signal: np.ndarray, dp_level: str, rle_mode_hint, sr: int):
    """
    Very lightweight rule of thumb:
      - Favor higher order for higher SR / higher DP
      - Always enable LP in Standard pipeline
    """
    dp_level = (dp_level or "").lower()
    hi_dp = dp_level in ("dp3", "dp4")
    order = 24 if (sr >= 44100 and hi_dp) else (12 if sr >= 32000 else 8)
    return True, order  # (use_lp, order)

def _raclp_auto_simple(signal: np.ndarray, level: int, rle_mode_hint, sr: int, dp_level: str):
    """
    Lightweight RACLP choice:
    - Small levels → smaller order; bigger levels → slightly bigger order
    - Always enable in Base-i
    """
    order = 2 if level <= 2 else (3 if level <= 4 else 4)
    return True, order  # (use_raclp, order)

# Wire them as our AUTO deciders
_decide_lp_auto_standard = _lp_auto_std_simple
_decide_raclp_auto = _raclp_auto_simple


# --- read JSON/MsgPack from compressed or plain container path ---
def _load_any_alak(path: str) -> dict:
    """
    Loads any of:
      .alak                         (JSON)
      .alak.bz2/.gz/.xz/.zst       (JSON, compressed)
      .alak.bin                     (MsgPack)
      .alak.bin.bz2/.gz/.xz/.zst   (MsgPack, compressed)
    Returns the parsed container dict.
    """
    p = Path(path)
    name = p.name.lower()
    raw = p.read_bytes()

    # Decompress if needed (by file extension)
    if name.endswith(".bz2"):
        import bz2
        raw = bz2.decompress(raw)
        name = name[:-4]
    elif name.endswith(".gz"):
        import gzip
        raw = gzip.decompress(raw)
        name = name[:-3]
    elif name.endswith(".xz") or name.endswith(".lzma"):
        import lzma
        raw = lzma.decompress(raw)
        name = name.rsplit(".", 1)[0]
    elif name.endswith(".zst"):
        try:
            import zstandard as zstd
        except Exception:
            raise RuntimeError("zstd not installed. Install with: pip install zstandard")
        raw = zstd.ZstdDecompressor().decompress(raw)
        name = name[:-4]

    # Decide JSON vs MsgPack based on inner name; fallback to sniff
    if name.endswith(".alak.bin"):
        if not HAS_MSGPACK:
            raise RuntimeError("MsgPack support missing. Install with: pip install 'msgpack>=1.0,<2'")
        return msgpack.unpackb(raw, raw=False)

    if name.endswith(".alak"):
        return json.loads(raw.decode("utf-8"))

    # Fallback: sniff by content
    head = raw.lstrip()[:1]
    if head == b"{":  # JSON object
        return json.loads(raw.decode("utf-8"))
    if not HAS_MSGPACK:
        raise RuntimeError("Unknown ALAK container and MsgPack not available.")
    return msgpack.unpackb(raw, raw=False)


# ---------- Player (sounddevice) ----------
class PlayerDialog(QtWidgets.QDialog):
    """
    Simple ALAK player:
      - Accepts a path to .alak / .alak.bin (optionally .bz2/.gz/.xz/.zst)
      - Loads + decodes audio
      - Plays via sounddevice
      - Shows duration + slider with live progress
    """
    def __init__(self, parent: QtWidgets.QWidget | None, path: str):
        super().__init__(parent)
        self.setWindowTitle(f"Play: {os.path.basename(path)}")
        self.resize(560, 180)

        self._y = None          # np.float32 mono
        self._sr = 44100
        self._pos = 0           # current frame
        self._lock = threading.Lock()
        self._stream = None
        self._playing = False

        # --- UI ---
        v = QtWidgets.QVBoxLayout(self)
        self.info_lbl = QtWidgets.QLabel("Loading…")
        v.addWidget(self.info_lbl)

        # Time + slider row
        h = QtWidgets.QHBoxLayout()
        self.time_lbl = QtWidgets.QLabel("00:00.000 / 00:00.000")
        self.time_lbl.setMinimumWidth(160)
        h.addWidget(self.time_lbl)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setSingleStep(1)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        h.addWidget(self.slider, 1)

        v.addLayout(h)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        btn_row.addStretch(1)
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        v.addLayout(btn_row)

        # Wire buttons
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)

        # Timer to update UI progress
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)  # ms
        self._timer.timeout.connect(self._tick)

        # Load & decode now
        self._load_and_decode(path)

    # ---------- loading ----------
    def _load_and_decode(self, path: str):
        try:
            # Prefer the library loader; it may handle plain .alak/.alak.bin
            try:
                cont = load_alak_file(path)
            except Exception:
                cont = _load_any_alak(path)

            payload = cont.get("compressed_data", {})
            sr = int(cont.get("sample_rate", 44100))
            y = decode_basei_sparse(payload)
            y = np.asarray(y)
            if np.iscomplexobj(y):
                y = y.real
            y = y.astype(np.float32, copy=False)
            # de-quantize if defensive path
            scale = float(payload.get("scale", 1.0)) or 1.0
            if (y.dtype.kind in "iu") or (np.nanmax(np.abs(y)) > 4.0 and scale > 1.0):
                y = y / scale

            # Normalize gently to avoid clipping
            mx = float(np.max(np.abs(y))) if y.size else 0.0
            if mx > 1.0:
                y = y / mx

            # Store
            self._y = y.reshape(-1)
            self._sr = sr
            self._pos = 0

            # UI init
            self.slider.setMaximum(max(0, len(self._y) - 1))
            self.slider.setEnabled(True)
            self._update_time_label()
            self.info_lbl.setText(f"{os.path.basename(path)}  |  {len(self._y)/self._sr:.3f}s @ {self._sr} Hz")

        except Exception:
            self.info_lbl.setText("Failed to load.")
            raise

    # ---------- playback ----------
    def _sd_callback(self, outdata, frames, time_info, status):
        # Called from audio thread
        if self._y is None:
            outdata.fill(0)
            return
        with self._lock:
            N = len(self._y)
            start = self._pos
            end = min(N, start + frames)
            chunk = self._y[start:end]
            outdata.fill(0)
            if chunk.size:
                outdata[:chunk.size, 0] = chunk
            self._pos = end

    def play(self):
        if not HAS_SD:
            QtWidgets.QMessageBox.critical(self, "Audio error", "sounddevice not installed.\n\npip install sounddevice")
            return
        if self._y is None or self._playing:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                callback=self._sd_callback,
                blocksize=0
            )
            self._stream.start()
            self._playing = True
            self._timer.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Audio error", str(e))

    def pause(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._playing = False
        self._timer.stop()

    def stop(self):
        self.pause()
        with self._lock:
            self._pos = 0
        self._update_slider_and_time()

    # ---------- slider/time ----------
    def _format_ms(self, ms: float) -> str:
        s = ms / 1000.0
        m, s2 = divmod(s, 60.0)
        return f"{int(m):02d}:{s2:06.3f}"

    def _update_time_label(self):
        if self._y is None:
            self.time_lbl.setText("00:00.000 / 00:00.000")
            return
        cur_ms = 1000.0 * (self._pos / max(1, self._sr))
        tot_ms = 1000.0 * (len(self._y) / max(1, self._sr))
        self.time_lbl.setText(f"{self._format_ms(cur_ms)} / {self._format_ms(tot_ms)}")

    def _update_slider_and_time(self):
        if self._y is None:
            return
        # Avoid recursive signals
        old = self.slider.blockSignals(True)
        self.slider.setValue(min(self._pos, self.slider.maximum()))
        self.slider.blockSignals(old)
        self._update_time_label()

    def _tick(self):
        # Stop timer if not playing but keep last position visible
        if self._y is not None and self._pos >= len(self._y):
            self.pause()
            return
        self._update_slider_and_time()

    def _on_slider_released(self):
        new_pos = int(self.slider.value())
        with self._lock:
            self._pos = new_pos

    def _on_slider_moved(self, value: int):
        # live scrub when paused
        if not self._playing:
            with self._lock:
                self._pos = int(value)

    # ---------- lifecycle ----------
    def closeEvent(self, ev: QtGui.QCloseEvent):
        try:
            self.pause()
        finally:
            super().closeEvent(ev)


# ---------- Main Window ----------
class AlakGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALAK Encoder (Beta UI)")
        self.resize(980, 720)

        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # --- File row ---
        file_row = QtWidgets.QHBoxLayout()
        self.in_edit = QtWidgets.QLineEdit()
        self.in_btn  = QtWidgets.QPushButton("Browse WAV…")
        self.out_dir_edit = QtWidgets.QLineEdit()
        self.out_dir_btn  = QtWidgets.QPushButton("Output Dir…")
        file_row.addWidget(QtWidgets.QLabel("Input WAV:"))
        file_row.addWidget(self.in_edit, 2)
        file_row.addWidget(self.in_btn)
        file_row.addSpacing(12)
        file_row.addWidget(QtWidgets.QLabel("Output dir:"))
        file_row.addWidget(self.out_dir_edit, 2)
        file_row.addWidget(self.out_dir_btn)
        root.addLayout(file_row)

        # --- Simple mode ---
        self.simple_chk = QtWidgets.QCheckBox("Simple mode (auto everything)")
        self.simple_chk.setChecked(True)
        root.addWidget(self.simple_chk)

        # --- Settings grid (2 tidy rows) ---
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # Row 0: DP | Pipeline | RLE
        grid.addWidget(QtWidgets.QLabel("DP:"), 0, 0)
        self.dp_combo = QtWidgets.QComboBox()
        self.dp_combo.addItems(["Auto", "raw", "dp1", "dp2", "dp3", "dp4"])
        self.dp_combo.setCurrentIndex(0)
        self.dp_combo.setMinimumWidth(180)
        grid.addWidget(self.dp_combo, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Pipeline:"), 0, 2)
        self.pipe_combo = QtWidgets.QComboBox()
        self.pipe_combo.addItems(["Auto", "Standard", "Base-i"])
        self.pipe_combo.setCurrentIndex(0)
        self.pipe_combo.setMinimumWidth(180)
        grid.addWidget(self.pipe_combo, 0, 3)

        grid.addWidget(QtWidgets.QLabel("RLE:"), 0, 4)
        self.rle_combo = QtWidgets.QComboBox()
        self.rle_combo.addItems(["Auto", "Off", "On"])
        self.rle_combo.setCurrentIndex(0)
        self.rle_combo.setMinimumWidth(120)
        grid.addWidget(self.rle_combo, 0, 5)

        # Row 1: Base-i Level (only when Base-i) | LP group (under Pipeline) | Order
        # Base-i widgets
        self.basei_level_lbl = QtWidgets.QLabel("Base-i Level (L):")
        self.basei_level = QtWidgets.QSpinBox()
        self.basei_level.setRange(1, 64)
        self.basei_level.setValue(1)
        self.basei_level.setMinimumWidth(80)
        grid.addWidget(self.basei_level_lbl, 1, 0)
        grid.addWidget(self.basei_level,     1, 1)

        # LP pages (Standard vs Base-i) stacked into one slot that spans cols 2..5
        
        # --- LP (Standard) page ---
        self.lp_std_page = QtWidgets.QWidget()
        self.lp_std_grid = QtWidgets.QGridLayout(self.lp_std_page)
        self.lp_std_grid.setContentsMargins(0, 0, 0, 0)
        self.lp_std_grid.setHorizontalSpacing(8)

        self.lp_std_lbl = QtWidgets.QLabel("LP (Standard):")
        self.lp_std_combo = QtWidgets.QComboBox()
        self.lp_std_combo.addItems(["Auto", "On", "Off"])
        self.lp_std_combo.setCurrentIndex(0)
        self.lp_std_ord_lbl = QtWidgets.QLabel("Order:")
        self.lp_std_order = QtWidgets.QSpinBox()
        self.lp_std_order.setRange(1, 64)
        self.lp_std_order.setValue(4)

        self.lp_std_grid.addWidget(self.lp_std_lbl,     0, 0)
        self.lp_std_grid.addWidget(self.lp_std_combo,   0, 1)
        self.lp_std_grid.addWidget(self.lp_std_ord_lbl, 0, 2)
        self.lp_std_grid.addWidget(self.lp_std_order,   0, 3)

        # lock widths + alignment so the row doesn't jump when "Order" hides
        self.lp_std_combo.setMinimumWidth(180)
        self.lp_std_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.lp_std_ord_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.lp_std_grid.setColumnMinimumWidth(0, 130)
        self.lp_std_grid.setColumnMinimumWidth(1, 180)
        self.lp_std_grid.setColumnMinimumWidth(2, 52)
        self.lp_std_grid.setColumnMinimumWidth(3, 80)
        self.lp_std_grid.setColumnStretch(1, 1)  # only the combo stretches


        # --- LP (Base-i / RACLP) page ---
        self.lp_bi_page = QtWidgets.QWidget()
        self.lp_bi_grid = QtWidgets.QGridLayout(self.lp_bi_page)
        self.lp_bi_grid.setContentsMargins(0, 0, 0, 0)
        self.lp_bi_grid.setHorizontalSpacing(8)

        self.lp_bi_lbl = QtWidgets.QLabel("LP (Base-i / RACLP):")
        self.lp_bi_combo = QtWidgets.QComboBox()
        self.lp_bi_combo.addItems(["Auto", "On", "Off"])
        self.lp_bi_combo.setCurrentIndex(0)
        self.lp_bi_ord_lbl = QtWidgets.QLabel("Order:")
        self.lp_bi_order = QtWidgets.QSpinBox()
        self.lp_bi_order.setRange(1, 64)
        self.lp_bi_order.setValue(4)

        self.lp_bi_grid.addWidget(self.lp_bi_lbl,     0, 0)
        self.lp_bi_grid.addWidget(self.lp_bi_combo,   0, 1)
        self.lp_bi_grid.addWidget(self.lp_bi_ord_lbl, 0, 2)
        self.lp_bi_grid.addWidget(self.lp_bi_order,   0, 3)

        # same anchoring as the Standard page
        self.lp_bi_combo.setMinimumWidth(180)
        self.lp_bi_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.lp_bi_ord_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.lp_bi_grid.setColumnMinimumWidth(0, 130)
        self.lp_bi_grid.setColumnMinimumWidth(1, 180)
        self.lp_bi_grid.setColumnMinimumWidth(2, 52)
        self.lp_bi_grid.setColumnMinimumWidth(3, 80)
        self.lp_bi_grid.setColumnStretch(1, 1)


        self.lp_stack = QtWidgets.QStackedWidget()
        self.lp_stack.addWidget(self.lp_std_page)  # index 0
        self.lp_stack.addWidget(self.lp_bi_page)   # index 1
        grid.addWidget(self.lp_stack, 1, 2, 1, 4)  # same slot as before


        # Keep columns breathable and aligned with row 0
        grid.setColumnStretch(1, 1)  # DP value
        grid.setColumnStretch(3, 1)  # Pipeline value
        grid.setColumnStretch(5, 1)  # RLE value
        root.addLayout(grid)

        # --- Compress & format ---
        comp_row = QtWidgets.QHBoxLayout()
        self.compress_chk = QtWidgets.QCheckBox("Compress output")
        self.compress_chk.setChecked(True)
        self.comp_method = QtWidgets.QComboBox()
        self.comp_method.addItems(["bz2", "gzip", "lzma/xz", "zstd"])
        self.comp_method.setCurrentIndex(0)  # default bz2
        self.binary_chk = QtWidgets.QCheckBox("Use binary container (MsgPack)")
        comp_row.addWidget(self.compress_chk)
        comp_row.addWidget(QtWidgets.QLabel("Method:"))
        comp_row.addWidget(self.comp_method)
        comp_row.addStretch(1)
        comp_row.addWidget(self.binary_chk)
        root.addLayout(comp_row)

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        self.play_btn  = QtWidgets.QPushButton("Play .alak…")
        self.encode_btn = QtWidgets.QPushButton("Encode")
        self.quit_btn   = QtWidgets.QPushButton("Quit")
        btn_row.addStretch(1)
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.encode_btn)
        btn_row.addWidget(self.quit_btn)
        root.addLayout(btn_row)

        # --- Log ---
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.log.setFont(mono)
        root.addWidget(self.log, 1)

        # Connections
        self.in_btn.clicked.connect(self.on_browse_wav)
        self.out_dir_btn.clicked.connect(self.on_browse_dir)
        self.quit_btn.clicked.connect(self.close)
        self.encode_btn.clicked.connect(self.on_encode_clicked)
        self.play_btn.clicked.connect(self.on_play_clicked)

        self.pipe_combo.currentIndexChanged.connect(self.sync_pipeline_deps)
        self.lp_std_combo.currentIndexChanged.connect(self.sync_lp_std_order_vis)
        self.lp_bi_combo.currentIndexChanged.connect(self.sync_lp_bi_order_vis)
        self.simple_chk.stateChanged.connect(self.sync_simple_mode)

        # Initial vis
        self.sync_simple_mode()
        self.sync_pipeline_deps()
        self.sync_lp_std_order_vis()
        self.sync_lp_bi_order_vis()

        self.log.appendPlainText(f"[info] GUI ready. Repo: {REPO}")

    # ---------- UI reactions ----------
    def on_browse_wav(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select WAV file", str(REPO), "WAV (*.wav)")
        if p:
            self.in_edit.setText(p)
            if not self.out_dir_edit.text().strip():
                self.out_dir_edit.setText(str(Path(p).parent))

    def on_browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", str(REPO))
        if d:
            self.out_dir_edit.setText(d)

    def sync_simple_mode(self):
        simple = self.simple_chk.isChecked()
        # Disable details in simple mode
        for w in (
            self.dp_combo, self.pipe_combo, self.rle_combo,
            self.basei_level, self.lp_std_combo, self.lp_std_order,
            self.lp_bi_combo, self.lp_bi_order, self.comp_method,
            self.binary_chk, self.compress_chk
        ):
            w.setEnabled(not simple)
        # Still show/hide groups properly
        self.sync_pipeline_deps()
        self.sync_lp_std_order_vis()
        self.sync_lp_bi_order_vis()

    def pipeline_choice(self) -> str:
        return {"Auto": "auto", "Standard": "std", "Base-i": "basei"}[self.pipe_combo.currentText()]

    def sync_pipeline_deps(self):
        kind = self.pipeline_choice()
        is_std = (kind == "std")
        is_bi  = (kind == "basei")

        # Base-i level only when Base-i
        self.basei_level_lbl.setVisible(is_bi)
        self.basei_level.setVisible(is_bi)

        # Pick the LP page under the Pipeline selector
        self.lp_stack.setCurrentIndex(1 if is_bi else 0)

        # Update per-page order visibility
        self.sync_lp_std_order_vis()
        self.sync_lp_bi_order_vis()

    def sync_lp_std_order_vis(self):
        show = (self.lp_stack.currentIndex() == 0 and self.lp_std_combo.currentText() == "On")
        self.lp_std_ord_lbl.setVisible(show)
        self.lp_std_order.setVisible(show)

    def sync_lp_bi_order_vis(self):
        show = (self.lp_stack.currentIndex() == 1 and self.lp_bi_combo.currentText() == "On")
        self.lp_bi_ord_lbl.setVisible(show)
        self.lp_bi_order.setVisible(show)

    # ---------- Encode ----------
    def on_encode_clicked(self):
        try:
            self._do_encode()
        except Exception as e:
            self.log.appendPlainText("❌ Encode failed:\n" + "".join(traceback.format_exception(e)))

    def _do_encode(self):
        wav_path = self.in_edit.text().strip()
        out_dir  = self.out_dir_edit.text().strip() or str(Path(wav_path).parent)
        if not wav_path or not Path(wav_path).exists():
            self.log.appendPlainText("❌ Please choose a valid WAV file.")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Load audio mono-float
        sig, sr = sf.read(wav_path, always_2d=False)
        if getattr(sig, "ndim", 1) > 1:
            sig = sig.mean(axis=1)
        sig = sig.astype(np.float32, copy=False)

        # Defaults from UI
        simple = self.simple_chk.isChecked()

        # DP
        dp_sel = self.dp_combo.currentText()
        if simple or dp_sel == "Auto":
            if dp_auto:
                picked = int(dp_auto(sig, sample_rate=int(sr)))
                dp = f"dp{picked}"
                self.log.appendPlainText(f"🤖 dp_auto → {dp}")
            else:
                dp = "dp4"
                self.log.appendPlainText("⚠️ dp_auto unavailable; using dp4.")
        else:
            dp = dp_sel

        # Pipeline
        if simple or self.pipe_combo.currentText() == "Auto":
            scheme = choose_pipeline_auto(sig, int(sr))
            pipe_label = "Base-i" if scheme == "basei_sparse_v1" else "Standard"
            self.log.appendPlainText(f"🤖 Pipeline AUTO → {pipe_label}")
        else:
            scheme = "basei_sparse_v1" if self.pipeline_choice() == "basei" \
                     else "time_intzz_v1"

        # RLE
        rle_txt = self.rle_combo.currentText()
        use_rle = None if (simple or rle_txt == "Auto") else (rle_txt == "On")

        # LP per pipeline
        use_lp = False; lp_order = 0
        use_raclp = False; raclp_order = 0
        L = int(self.basei_level.value())

        if scheme == "time_intzz_v1":
            mode = "Auto" if simple else self.lp_std_combo.currentText()
            if mode == "On":
                use_lp = True
                lp_order = int(self.lp_std_order.value())
            elif mode == "Off":
                use_lp = False
            else:  # Auto
                if _decide_lp_auto_standard:
                    use_lp, chosen = _decide_lp_auto_standard(sig, dp_level=dp, rle_mode_hint=use_rle, sr=int(sr))
                    lp_order = int(chosen if use_lp else 0)
                    self.log.appendPlainText(f"🤖 LP AUTO (Standard) → {'on' if use_lp else 'off'}"
                                             + (f' (order={lp_order})' if use_lp else ''))
                else:
                    use_lp = True; lp_order = 4
                    self.log.appendPlainText("ℹ️ LP AUTO fallback → on (order=4)")
        else:
            mode = "Auto" if simple else self.lp_bi_combo.currentText()
            if mode == "On":
                use_raclp = True
                raclp_order = int(self.lp_bi_order.value())
            elif mode == "Off":
                use_raclp = False
            else:  # Auto
                if _decide_raclp_auto:
                    use_raclp, chosen = _decide_raclp_auto(sig, level=L, rle_mode_hint=use_rle, sr=int(sr), dp_level=dp)
                    raclp_order = int(chosen if use_raclp else 0)
                    self.log.appendPlainText(f"🤖 LP AUTO (Base-i/RACLP) → {'on' if use_raclp else 'off'}"
                                             + (f' (order={raclp_order})' if use_raclp else ''))
                else:
                    use_raclp = True; raclp_order = 4
                    self.log.appendPlainText("ℹ️ RACLP AUTO fallback → on (order=4)")

        # Do encode
        t0 = time.time()
        enc_kwargs = dict(
            signal=sig,
            dp=dp,
            scheme=scheme,
            use_rle=use_rle,
            prune_counts=0,
            level=L,
            use_raclp=use_raclp,
            lp_order=(raclp_order if scheme == "basei_sparse_v1" else lp_order),
            use_lp=(use_lp if scheme == "time_intzz_v1" else False),
        )
        encoded = encode_basei_sparse(**enc_kwargs)
        dt = (time.time() - t0) * 1000.0
        self.log.appendPlainText(f"✅ Encoded in {dt:.1f} ms")

        # Build effective tags for filename
        dp_tag  = dp or "raw"
        pipe_tag = "basei" if scheme == "basei_sparse_v1" else "std"
        level_tag = f".L{L}" if scheme == "basei_sparse_v1" else ""
        lp_tag, rle_tag = effective_tags_from_payload(encoded, scheme)
        base = Path(wav_path).stem

        use_binary = self.binary_chk.isChecked()
        ext = ".alak.bin" if use_binary else ".alak"
        auto_name = f"{base}.{dp_tag}.{pipe_tag}{level_tag}.{lp_tag}.{rle_tag}{ext}"
        out_path = Path(out_dir) / auto_name

        # Save
        try:
            save_alak_file(
                path=str(out_path),
                sample_rate=int(sr),
                original_length=len(sig),
                dp=dp,
                compressed_data=encoded,
                binary=bool(use_binary)
            )
            if use_binary:
                self.log.appendPlainText("💾 Wrote binary MsgPack container.")
        except TypeError:
            # Older saver without 'binary' kwarg
            save_alak_file(
                path=str(out_path),
                sample_rate=int(sr),
                original_length=len(sig),
                dp=dp,
                compressed_data=encoded,
            )
            if use_binary:
                self.log.appendPlainText("ℹ️ Binary requested, but saver does not support it. Wrote JSON; using .alak.bin extension for clarity.")

        self.log.appendPlainText(f"💾 Saved: {out_path.name}  ({human_bytes(out_path.stat().st_size)})")

        # Compress if requested or simple mode
        cmp_path = None
        if self.compress_chk.isChecked() or self.simple_chk.isChecked():
            method_txt = self.comp_method.currentText()
            method = {"bz2":"bz2", "gzip":"gzip", "lzma/xz":"lzma", "zstd":"zstd"}[method_txt]
            try:
                cmp_path = compress_alak_path(str(out_path), method)
                self.log.appendPlainText(f"📦 Compressed → {Path(cmp_path).name}  ({human_bytes(Path(cmp_path).stat().st_size)})")
            except Exception as e:
                self.log.appendPlainText(f"❌ Compression failed: {e}")

        # Basic report
        wav_size = Path(wav_path).stat().st_size
        if cmp_path:
            comp_size = Path(cmp_path).stat().st_size
            red2 = (1 - comp_size / wav_size) * 100.0 if wav_size else 0.0
            ratio2 = (wav_size / comp_size) if comp_size else float("inf")
            self.log.appendPlainText(f"\n🎯 End-to-End Reduction (wav→cmp): {red2:.1f}%   Ratio: {ratio2:.1f}x smaller\n")

        # Quick quality metrics
        try:
            recon = decode_basei_sparse(encoded)[: len(sig)]
            recon = np.asarray(recon)
            if np.iscomplexobj(recon):
                recon = recon.real
            recon = recon.astype(np.float32, copy=False)
            # de-quantize if decoder returned ints (defensive)
            scale = float(encoded.get("scale", 1.0)) or 1.0
            if (recon.dtype.kind in "iu") or (np.nanmax(np.abs(recon)) > 4.0 and scale > 1.0):
                recon = recon / scale

            err = sig[: len(recon)] - recon
            mse = float(np.mean(err * err))
            mae = float(np.mean(np.abs(err)))
            sig_pow = float(np.mean(sig[: len(recon)] ** 2)) + 1e-12
            snr_db = 10.0 * math.log10(sig_pow / (mse + 1e-12))
            self.log.appendPlainText("📊 Quick Metrics")
            self.log.appendPlainText(f"   MSE={mse:.8f}  MAE={mae:.6f}  SNR={snr_db:.2f} dB\n")
        except Exception as e:
            self.log.appendPlainText(f"(metrics skipped: {e})")

    # ---------- Play ----------
    def on_play_clicked(self):
        start_dir = self.out_dir_edit.text().strip() or str(REPO)
        filters = (
            "ALAK containers (*.alak *.alak.bz2 *.alak.gz *.alak.xz *.alak.zst "
            "*.alak.bin *.alak.bin.bz2 *.alak.bin.gz *.alak.bin.xz *.alak.bin.zst);;"
            "All files (*)"
        )
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open ALAK to play", start_dir, filters)
        if not p:
            return
        try:
            dlg = PlayerDialog(self, p)
            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Play error", str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = AlakGUI()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
