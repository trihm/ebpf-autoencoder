#!/usr/bin/env python3
"""
parity_check.py — bang chung rang feature LIVE == feature BATCH.

Replay file log qua StreamAggregator, roi so tung o voi ket qua
aggregate_features.py chay batch tren cung file. Khac 0 o => khong co
train/serve skew. Chay truoc khi tin bat ky so nao cua detector live.

  python3 parity_check.py \
      --hubble   /tmp/ebpf-logs/cilium-network.json \
      --tetragon /tmp/ebpf-logs/tetragon-events.json
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import aggregate_features as agg
from stream_features import StreamAggregator, parse_hubble_obj, parse_tetragon_obj

KEY = ("flow_key", "window_start")
T0 = time.time()


def log(msg):
    """In tien do kem thoi gian troi qua. flush de thay ngay khi pipe/tee."""
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def snapshot(paths, tmpdir):
    """Dong bang log truoc khi so sanh.

    parity_check mo file 4 lan (2 cho batch, 2 cho live). Neu sensor van dang
    ghi, nhanh live doc sau se thay them event o duoi -> bao 'chi co o live'
    va lech duration/flow_count o cua so cuoi, DU pipeline hoan toan dung.
    Chep ra tmp va bo dong cuoi (co the dang duoc ghi do dang)."""
    out = []
    for i, p in enumerate(paths):
        dst = os.path.join(tmpdir, f"{i:02d}_{os.path.basename(p)}")
        with open(p, "r", errors="replace") as src:
            lines = src.readlines()
        if lines and not lines[-1].endswith("\n"):
            lines.pop()                      # dong viet do dang
        elif lines:
            lines.pop()                      # dong cuoi co the vua duoc ghi
        with open(dst, "w") as fh:
            fh.writelines(lines)
        out.append(dst)
        log(f"  snapshot {os.path.basename(p)}: {len(lines)} dong")
    return out


def index_rows(rows):
    return {(r["flow_key"], int(r["window_start"])): r for r in rows}


def compare(name, batch_rows, live_rows, cols, tol=1e-9):
    b, l = index_rows(batch_rows), index_rows(live_rows)
    only_b = sorted(set(b) - set(l))
    only_l = sorted(set(l) - set(b))
    diffs = []
    for k in sorted(set(b) & set(l)):
        for c in cols:
            vb, vl = float(b[k][c]), float(l[k][c])
            if abs(vb - vl) > tol:
                diffs.append((k, c, vb, vl))
    print(f"\n=== {name} ===")
    print(f"batch: {len(b)} dong | live: {len(l)} dong | chung: {len(set(b) & set(l))}")
    print(f"chi co o batch: {len(only_b)} | chi co o live: {len(only_l)} | o lech: {len(diffs)}")
    for k in only_b[:5]:
        print(f"  [batch-only] {k}")
    for k in only_l[:5]:
        print(f"  [live-only ] {k}")
    for d in diffs[:10]:
        print(f"  [lech] {d[0]} {d[1]}: batch={d[2]} live={d[3]}")
    return not only_b and not only_l and not diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubble", required=True)
    ap.add_argument("--tetragon", required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rolling", type=int, default=60)
    ap.add_argument("--no-snapshot", action="store_true",
                    help="doc thang file goc (chi dung khi log da dung han)")
    args = ap.parse_args()

    hub_paths = agg.expand_paths(args.hubble)
    tet_paths = agg.expand_paths(args.tetragon)
    log(f"file hubble: {hub_paths}")
    log(f"file tetragon: {tet_paths}")

    tmpdir = None
    if not args.no_snapshot:
        tmpdir = tempfile.mkdtemp(prefix="parity-")
        log("dong bang log (tranh sensor ghi them trong luc chay)...")
        hub_paths = snapshot(hub_paths, tmpdir)
        tet_paths = snapshot(tet_paths, tmpdir)

    # --- batch ---
    log("doc log Hubble (batch)...")
    net_recs, skip_n = agg.load_hubble(hub_paths)
    log(f"  {len(net_recs)} record network (bo qua {skip_n})")
    log("doc log Tetragon (batch)...")
    proc_recs, skip_p, _, _ = agg.load_tetragon(tet_paths)
    log(f"  {len(proc_recs)} record process (bo qua {skip_p})")

    if net_recs:
        span = (net_recs[-1]["t"] - net_recs[0]["t"]) / 60.0
        log(f"  log trai dai {span:.1f} phut ~ {int(span * 60 / args.window)} cua so")
    log("aggregate network (batch)...")
    batch_net = agg.aggregate_network(net_recs, args.window, args.rolling)
    log(f"  {len(batch_net)} dong")
    log("aggregate process (batch)...")
    batch_proc = agg.aggregate_process(proc_recs, args.window, args.rolling)
    log(f"  {len(batch_proc)} dong")

    # --- live (replay) ---
    log("parse lai log cho duong live...")
    events = []
    for obj in agg.iter_json_lines(hub_paths):
        r = parse_hubble_obj(obj)
        if r:
            events.append(("net", r))
    for obj in agg.iter_json_lines(tet_paths):
        r = parse_tetragon_obj(obj)
        if r:
            events.append(("proc", r))
    events.sort(key=lambda e: e[1]["t"])
    total = len(events)
    log(f"  {total} event, bat dau replay")

    sa = StreamAggregator(window=args.window, rolling=args.rolling, grace=2.0)
    live_net, live_proc = [], []
    step = max(1, total // 20)          # in khoang 20 lan
    n_win = 0
    for i, (kind, rec) in enumerate(events, 1):
        (sa.add_network if kind == "net" else sa.add_process)(rec)
        for _, nr, pr in sa.pop_ready(now=0):  # now=0 -> watermark = thoi gian event lon nhat
            live_net += nr
            live_proc += pr
            n_win += 1
        if i % step == 0 or i == total:
            log(f"  replay {i}/{total} event ({100 * i // total}%) "
                f"| {n_win} cua so | buffer {len(sa.net_recs)}+{len(sa.proc_recs)}")
    for _, nr, pr in sa.flush():
        live_net += nr
        live_proc += pr
        n_win += 1
    log(f"replay xong: {n_win} cua so, {len(live_net)} dong net, {len(live_proc)} dong proc")

    log("so sanh batch vs live...")
    ok_net = compare("network", batch_net, live_net, agg.NETWORK_FEATURES)
    ok_proc = compare("process", batch_proc, live_proc, agg.PROCESS_FEATURES)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nevent den tre bi bo: {sa.dropped_late}")
    print(f"tong thoi gian: {time.time() - T0:.1f}s")
    print("\nKET QUA: " + ("PARITY OK" if ok_net and ok_proc else "CO LECH — dung lai va sua"))
    return 0 if (ok_net and ok_proc) else 1


if __name__ == "__main__":
    sys.exit(main())
