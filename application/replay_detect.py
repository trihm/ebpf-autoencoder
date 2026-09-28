#!/usr/bin/env python3
"""
replay_detect.py — chay TOAN BO duong di live (stream -> score -> fusion) tren
file log attack da co, nhung offline va co nhan.

Dung de tra loi cau hoi: "detector live co cho ra dung con so trong
ket_qua_chuong4.md khong?". Neu recall per-attack lech nhieu so voi
evaluate_v3.py thi loi nam o duong di live, khong phai o model.

  python3 replay_detect.py \
      --hubble /tmp/ebpf-logs/attack-network.json \
      --tetragon /tmp/ebpf-logs/attack-tetragon.json \
      --labels labels.csv --model-dir /tmp/ebpf-logs/models \
      --out live_scored.csv

labels.csv: cot attack,start,end (RFC3339) — giong file dua cho evaluate_v3.py.
"""
import argparse
import csv
from datetime import datetime

import aggregate_features as agg
from detector_core import (
    NETWORK_FEATURES, PROCESS_FEATURES, Branch, fuse_rows,
)
from stream_features import StreamAggregator, parse_hubble_obj, parse_tetragon_obj


def rfc3339_epoch(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def load_labels(path):
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append((row["attack"], rfc3339_epoch(row["start"]),
                        rfc3339_epoch(row["end"])))
    return out


def label_of(win, window, labels):
    a, b = win, win + window
    for name, s, e in labels:
        if a < e and b > s:      # giao nhau
            return name
    return "benign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubble", required=True)
    ap.add_argument("--tetragon", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model-dir", default="/tmp/ebpf-logs/models")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rolling", type=int, default=60)
    ap.add_argument("--fusion", default="max", choices=["max", "or"])
    ap.add_argument("--out", default="live_scored.csv")
    args = ap.parse_args()

    md = args.model_dir
    net_b = Branch("network", f"{md}/net_model.pt", f"{md}/net_scaler.pkl",
                   f"{md}/net_thr.txt", NETWORK_FEATURES)
    proc_b = Branch("process", f"{md}/proc_model.pt", f"{md}/proc_scaler.pkl",
                    f"{md}/proc_thr.txt", PROCESS_FEATURES)
    labels = load_labels(args.labels)

    events = []
    for obj in agg.iter_json_lines(agg.expand_paths(args.hubble)):
        r = parse_hubble_obj(obj)
        if r:
            events.append(("net", r))
    for obj in agg.iter_json_lines(agg.expand_paths(args.tetragon)):
        r = parse_tetragon_obj(obj)
        if r:
            events.append(("proc", r))
    events.sort(key=lambda e: e[1]["t"])

    sa = StreamAggregator(window=args.window, rolling=args.rolling)
    scored = []

    def handle(batch):
        for win, net_rows, proc_rows in batch:
            rows = fuse_rows(net_rows, proc_rows, net_b, proc_b, mode=args.fusion)
            lab = label_of(win, args.window, labels)
            scored.append({
                "window_start": win,
                "label": lab,
                "max_score": max((r["score"] for r in rows), default=0.0),
                "n_flows": len(rows),
                "n_alerts": sum(1 for r in rows if r["alert"]),
                "alert": any(r["alert"] for r in rows),
                "top": ",".join(
                    f["feature"] for r in rows if r["alert"] for f in r["top"][:1]
                )[:120],
            })

    for kind, rec in events:
        (sa.add_network if kind == "net" else sa.add_process)(rec)
        handle(sa.pop_ready(now=0))
    handle(sa.flush())

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)

    per = {}
    for s in scored:
        d = per.setdefault(s["label"], [0, 0])
        d[0] += 1
        d[1] += int(s["alert"])
    print(f"{'nhan':<18}{'bin':>6}{'alert':>8}{'ty le':>8}")
    for lab, (n, a) in sorted(per.items()):
        print(f"{lab:<18}{n:>6}{a:>8}{a / n:>8.3f}")
    print(f"\n-> {args.out}  (benign = FPR theo time-bin, con lai = recall)")


if __name__ == "__main__":
    main()
