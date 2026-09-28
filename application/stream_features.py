#!/usr/bin/env python3
"""
stream_features.py — bien pipeline batch thanh streaming MA KHONG viet lai
logic feature.

Y tuong: giu mot buffer event trong RAM dai (rolling + window) giay. Khi cua so
W du chin (watermark >= W + window + grace), goi THANG aggregate_network() /
aggregate_process() cua aggregate_features.py tren buffer, roi loc lay dung cac
dong co window_start == W.

Vi rolling-60s cua cua so W chi phu thuoc event trong [W+window-rolling, W+window),
va buffer luon chua tron khoang do, ket qua GIONG HET ban batch theo dinh nghia.
Khong co train/serve skew ve feature.

Phu thuoc: aggregate_features.py phai nam trong PYTHONPATH.
"""
import json
import os
import time
from collections import deque

import aggregate_features as agg


# ---------------------------------------------------------------------------
# Parse tung dong JSON (tach ra tu load_hubble / load_tetragon)
# ---------------------------------------------------------------------------

def parse_hubble_obj(obj):
    """Mot dong NDJSON Hubble -> record chuan hoa, hoac None neu bo qua.
    Than ham phai GIONG HET vong lap trong agg.load_hubble()."""
    f = obj.get("flow")
    if not f:
        return None
    ipinfo = f.get("IP")
    l4 = f.get("l4") or {}
    if not ipinfo or not l4:
        return None
    proto = next(iter(l4.keys()), None)
    if proto is None:
        return None
    l4d = l4[proto] or {}
    try:
        t = agg.parse_time(f["time"])
    except Exception:
        return None
    return {
        "t": t,
        "src": ipinfo.get("source", ""),
        "dst": ipinfo.get("destination", ""),
        "sport": l4d.get("source_port", 0),
        "dport": l4d.get("destination_port", 0),
        "proto": proto,
        "verdict": f.get("verdict", ""),
        "direction": f.get("traffic_direction", ""),
        "is_reply": bool(f.get("is_reply", False)),
        "l7": f.get("l7") is not None,
        "src_ns": (f.get("source") or {}).get("namespace", ""),
        "dst_ns": (f.get("destination") or {}).get("namespace", ""),
    }


def parse_tetragon_obj(obj):
    """Mot dong NDJSON Tetragon -> record chuan hoa, hoac None."""
    pk = obj.get("process_kprobe")
    if not pk:
        return None
    args = pk.get("args") or []
    sock = None
    nbytes = None
    for a in args:
        if "sock_arg" in a:
            sock = a["sock_arg"]
        elif "int_arg" in a:
            nbytes = a["int_arg"]
    if not sock:
        return None
    try:
        t = agg.parse_time(obj["time"])
    except Exception:
        return None
    proc = pk.get("process") or {}
    return {
        "t": t,
        "src": sock.get("saddr", ""),
        "dst": sock.get("daddr", ""),
        "sport": sock.get("sport", 0),
        "dport": sock.get("dport", 0),
        "proto": sock.get("protocol", "").replace("IPPROTO_", ""),
        "fn": pk.get("function_name", ""),
        "bytes": nbytes if nbytes is not None else 0,
        "binary": proc.get("binary", ""),
        "uid": proc.get("uid", -1),
    }


# ---------------------------------------------------------------------------
# Cua so truot theo watermark
# ---------------------------------------------------------------------------

class StreamAggregator:
    """Nhan record le, tra ve (window_start, net_rows, proc_rows) khi cua so chin."""

    def __init__(self, window=5, rolling=60, grace=2.0, clock_skew=3.0):
        self.window = window
        self.rolling = rolling
        self.grace = grace
        self.clock_skew = clock_skew
        self.net_recs = deque()
        self.proc_recs = deque()
        self.max_event_t = 0.0
        self.next_window = None   # cua so nho nhat chua dong
        self.dropped_late = 0

    def add_network(self, rec):
        self._add(self.net_recs, rec)

    def add_process(self, rec):
        self._add(self.proc_recs, rec)

    def _add(self, buf, rec):
        t = rec["t"]
        if self.next_window is not None and t < self.next_window:
            self.dropped_late += 1   # event den sau khi cua so da dong
            return
        buf.append(rec)
        if t > self.max_event_t:
            self.max_event_t = t
        if self.next_window is None:
            self.next_window = int(t // self.window) * self.window

    def watermark(self, now=None):
        """Lay max(thoi gian event lon nhat, gio he thong - clock_skew).
        Nhanh thu hai giup dong cua so khi traffic lang, mien la sensor va
        detector chung dong ho (cung host/cluster)."""
        now = time.time() if now is None else now
        return max(self.max_event_t, now - self.clock_skew)

    def pop_ready(self, now=None):
        """Tra ve list cac cua so da chin, moi phan tu (win, net_rows, proc_rows)."""
        out = []
        wm = self.watermark(now)
        while (
            self.next_window is not None
            and wm >= self.next_window + self.window + self.grace
        ):
            win = self.next_window
            out.append((win, *self._compute(win)))
            self.next_window = win + self.window
            self._prune()
        return out

    def flush(self):
        """Dong TAT CA cua so con lai (dung khi replay file, khong dung khi chay live)."""
        out = []
        while self.next_window is not None and self.next_window <= self.max_event_t:
            win = self.next_window
            out.append((win, *self._compute(win)))
            self.next_window = win + self.window
            self._prune()
        return out

    def _compute(self, win):
        net_rows = [
            r for r in agg.aggregate_network(list(self.net_recs), self.window, self.rolling)
            if r["window_start"] == win
        ]
        proc_rows = [
            r for r in agg.aggregate_process(list(self.proc_recs), self.window, self.rolling)
            if r["window_start"] == win
        ]
        return net_rows, proc_rows

    def _prune(self):
        """Giu du lich su cho rolling cua cua so ke tiep: [next+window-rolling, ...)."""
        keep_from = self.next_window + self.window - self.rolling
        for buf in (self.net_recs, self.proc_recs):
            while buf and buf[0]["t"] < keep_from:
                buf.popleft()

    def stats(self):
        return {
            "buffered_net": len(self.net_recs),
            "buffered_proc": len(self.proc_recs),
            "dropped_late": self.dropped_late,
            "max_event_t": self.max_event_t,
            "next_window": self.next_window,
        }


# ---------------------------------------------------------------------------
# Doc file NDJSON dang duoc ghi (tail -F), chiu duoc rotate/truncate
# ---------------------------------------------------------------------------

class LogTailer:
    def __init__(self, path, from_start=False):
        self.path = path
        self.from_start = from_start
        self.fh = None
        self.inode = None
        self.buf = ""

    def _open(self):
        if not os.path.exists(self.path):
            return False
        fh = open(self.path, "r", errors="replace")
        if not self.from_start:
            fh.seek(0, os.SEEK_END)
        self.fh = fh
        self.inode = os.fstat(fh.fileno()).st_ino
        return True

    def read_objects(self, max_lines=20000):
        """Doc cac dong moi, yield dict JSON. Khong block."""
        if self.fh is None and not self._open():
            return
        try:
            st = os.stat(self.path)
            if st.st_ino != self.inode or st.st_size < self.fh.tell():
                self.fh.close()
                self.from_start = True   # file moi -> doc tu dau
                if not self._open():
                    return
        except FileNotFoundError:
            return

        for _ in range(max_lines):
            line = self.fh.readline()
            if not line:
                break
            if not line.endswith("\n"):
                self.buf += line        # dong viet do dang
                break
            line = (self.buf + line).strip()
            self.buf = ""
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
