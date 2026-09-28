#!/usr/bin/env python3
"""
Feature aggregation cho DUAL-autoencoder (late fusion).

Nguon:
  - cilium-network.json  (Hubble flow, L3/L4)  -> bo dac trung QUAN HE (network branch)
  - tetragon-events.json (kprobe tcp_*)        -> bo dac trung VOLUME  (process branch)

Don vi flow: 5-tuple da chuan hoa hai chieu + cua so 5 giay.
Moi cua so cung sinh 2 dac trung rolling per-source 60s (port/ip scan).

Output: HAI file CSV RIENG, cung khoa (flow_key, window_start).
Khong join o day — moi nhanh train mot autoencoder rieng, fusion.py se
outer-join hai diem so sau khi cham. Do la diem khac voi ban single-AE.
  - features_network.csv   (15 feature)
  - features_process.csv   (18 feature, KHONG co connect_count vi tcp_connect
                            khong duoc hook trong TracingPolicy)

Chay:
  python3 aggregate_features.py \
      --hubble   /tmp/ebpf-logs/cilium-network.json \
      --tetragon /tmp/ebpf-logs/tetragon-events.json \
      --outdir   /tmp/ebpf-logs/features \
      --window   5 --rolling 60
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime

# ----------------------------------------------------------------------------
# Tien ich chung
# ----------------------------------------------------------------------------

def parse_time(ts):
    """RFC3339 nano -> epoch float. Cat nano ve micro cho fromisoformat."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if "." in ts:
        head, tail = ts.split(".", 1)
        for sign in ("+", "-"):
            idx = tail.find(sign)
            if idx != -1:
                frac, off = tail[:idx], tail[idx:]
                break
        else:
            frac, off = tail, ""
        frac = frac[:6]  # micro
        ts = f"{head}.{frac}{off}"
    return datetime.fromisoformat(ts).timestamp()


def normalize_ip(ip):
    """Strip IPv4-mapped IPv6 prefix: '::ffff:10.0.0.45' -> '10.0.0.45'.
    Tranh phan manh flow va sai rolling per-source count."""
    if not ip:
        return ip
    low = ip.lower()
    if low.startswith("::ffff:"):
        rest = ip[7:]
        if rest.count(".") == 3:
            return rest
    return ip


def is_junk_endpoint(ip, proto):
    """Loc flow rac: 0.0.0.0 / rong / proto None."""
    if not ip or ip in ("0.0.0.0", "::", "::ffff:0.0.0.0"):
        return True
    if not proto:
        return True
    return False


def canonical_flow_key(ip_a, port_a, ip_b, port_b, proto):
    """5-tuple hai chieu -> khoa on dinh. Endpoint port cao = 'client/forward'."""
    ip_a = normalize_ip(ip_a)
    ip_b = normalize_ip(ip_b)
    a = (ip_a, port_a)
    b = (ip_b, port_b)
    if port_a >= port_b:
        fwd_src, fwd_dst = a, b
    else:
        fwd_src, fwd_dst = b, a
    key = (fwd_src[0], fwd_src[1], fwd_dst[0], fwd_dst[1], proto)
    return key, fwd_src, fwd_dst


def safe_std(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def iter_json_lines(paths):
    """Doc nhieu file NDJSON, bo qua dong hong."""
    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def expand_paths(pattern):
    """Cho phep glob: cilium-network.json* de bat ca hau to .19700101."""
    matches = sorted(glob.glob(pattern))
    if not matches and os.path.exists(pattern):
        matches = [pattern]
    return matches


# ----------------------------------------------------------------------------
# NHANH NETWORK (Hubble)
# ----------------------------------------------------------------------------

NETWORK_FEATURES = [
    "flow_count", "duration",
    "tcp_ratio", "udp_ratio",
    "egress_ratio", "is_reply_ratio",
    "dropped_ratio", "l7_present_ratio", "dns_ratio",
    "distinct_dst_ports", "distinct_dst_ips", "distinct_dst_ns",
    "cross_ns_ratio",
    "distinct_dst_ports_60s", "distinct_dst_ips_60s",
]


def load_hubble(paths):
    """Tra ve list record da chuan hoa, sap theo time."""
    recs = []
    skipped = 0
    for obj in iter_json_lines(paths):
        f = obj.get("flow")
        if not f:
            skipped += 1
            continue
        ipinfo = f.get("IP")
        l4 = f.get("l4") or {}
        if not ipinfo or not l4:
            skipped += 1
            continue
        proto = next(iter(l4.keys()), None)
        if proto is None:
            skipped += 1
            continue
        l4d = l4[proto] or {}
        sport = l4d.get("source_port", 0)
        dport = l4d.get("destination_port", 0)
        try:
            t = parse_time(f["time"])
        except Exception:
            skipped += 1
            continue
        recs.append({
            "t": t,
            "src": ipinfo.get("source", ""),
            "dst": ipinfo.get("destination", ""),
            "sport": sport,
            "dport": dport,
            "proto": proto,            # "TCP" / "UDP" / "ICMPv4" ...
            "verdict": f.get("verdict", ""),
            "direction": f.get("traffic_direction", ""),
            "is_reply": bool(f.get("is_reply", False)),
            "l7": f.get("l7") is not None,
            "src_ns": (f.get("source") or {}).get("namespace", ""),
            "dst_ns": (f.get("destination") or {}).get("namespace", ""),
        })
    recs.sort(key=lambda r: r["t"])
    return recs, skipped


def _rolling_per_source(recs, window, rolling, raw=True):
    """Precompute rolling per-source-IP count.
    Tra ve dict: (src_ip, win_start) -> (distinct_dst_ports_60s, distinct_dst_ips_60s).

    Tinh DOC LAP voi bucket flow_key: gom TAT CA request (khong reply) cua cung
    mot source IP theo thoi gian, roi voi moi window dem so port/ip dich rieng biet
    trong 'rolling' giay gan nhat. Day la cach dung de bat port scan / lateral move:
    nmap tao nhieu flow_key khac nhau (ephemeral src port) nhung cung 1 src IP.

    raw=True  : recs la ban ghi THO (Hubble) — co sport/dport/proto, can chuan hoa
                chieu ngay tai day (nhanh network truyen recs goc).
    raw=False : recs DA chuan hoa san — moi phan tu co san src/dport/dst la huong
                forward that su (nhanh process da lam o aggregate_process).
    """
    # src_ip -> list (t, dport, dst_ip), da sap theo thoi gian (recs da sorted)
    per_src = defaultdict(list)
    windows_of_src = defaultdict(set)
    for r in recs:
        if r.get("is_reply"):
            continue
        if raw:
            # SUA BUG DEM EPHEMERAL: khong tin r["dport"] tho cua Hubble. Hubble ghi
            # ca hai chieu; khi ban ghi la chieu server->client (khong phai luc nao
            # cung co is_reply), r["dport"] chinh la ephemeral SOURCE port cua client
            # (33xxx, 55xxx...). Dem chung se phong distinct_dst_ports_60s len hang
            # tram cho slowloris/brute (von chi nham 1 cong). Chuan hoa chieu bang
            # cung logic canonical_flow_key: endpoint port THAP = cong dich that su.
            key, fwd_src, fwd_dst = canonical_flow_key(
                r["src"], r["sport"], r["dst"], r["dport"], r["proto"])
            real_src = fwd_src[0]          # IP nguoi khoi tao (port cao)
            real_dport = fwd_dst[1]        # cong dich that (port thap)
            real_dst = fwd_dst[0]
        else:
            # da chuan hoa san (proc_events): dung thang
            real_src = normalize_ip(r["src"])
            real_dport = r["dport"]
            real_dst = normalize_ip(r["dst"])
        per_src[real_src].append((r["t"], real_dport, real_dst))
        win = int(r["t"] // window) * window
        windows_of_src[real_src].add(win)

    out = {}
    for src, events in per_src.items():
        events.sort()
        for win in sorted(windows_of_src[src]):
            wend = win + window
            lo = wend - rolling
            # dem port/ip dich cua request trong [lo, wend)
            ports = set()
            ips = set()
            for t, dp, dip in events:
                if lo <= t < wend:
                    ports.add(dp)
                    ips.add(dip)
                elif t >= wend:
                    break
            out[(src, win)] = (len(ports), len(ips))
    return out


def aggregate_network(recs, window, rolling):
    """Gop theo (flow_key, window). Rolling 60s tinh per source-ip (precompute rieng)."""
    roll = _rolling_per_source(recs, window, rolling)
    buckets = defaultdict(list)

    for r in recs:
        if is_junk_endpoint(r["src"], r["proto"]) or is_junk_endpoint(r["dst"], r["proto"]):
            continue
        key, fwd_src, fwd_dst = canonical_flow_key(
            r["src"], r["sport"], r["dst"], r["dport"], r["proto"])
        win = int(r["t"] // window) * window
        buckets[(key, win)].append((r, fwd_src, fwd_dst))

    rows = []
    for (key, win), items in sorted(buckets.items(), key=lambda kv: kv[0][1]):
        rs = [it[0] for it in items]
        n = len(rs)
        times = [r["t"] for r in rs]
        duration = max(times) - min(times) if n > 1 else 0.0

        tcp = sum(1 for r in rs if r["proto"] == "TCP")
        udp = sum(1 for r in rs if r["proto"] == "UDP")
        egress = sum(1 for r in rs if r["direction"] == "EGRESS")
        reply = sum(1 for r in rs if r["is_reply"])
        dropped = sum(1 for r in rs if r["verdict"] == "DROPPED")
        l7 = sum(1 for r in rs if r["l7"])
        dns = sum(1 for r in rs if r["dport"] == 53 or r["sport"] == 53)

        # Port DICH that su cua request. Dung chieu chuan hoa (cong thap = dich),
        # KHONG tin r["dport"] tho (co the la ephemeral source port khi Hubble ghi
        # chieu server->client). Cung logic voi _rolling_per_source da sua.
        dst_ports = set()
        dst_ips = set()
        real_src_set = set()
        for r in rs:
            k, fsrc, fdst = canonical_flow_key(
                r["src"], r["sport"], r["dst"], r["dport"], r["proto"])
            dst_ports.add(fdst[1])
            dst_ips.add(fdst[0])
            real_src_set.add(fsrc[0])
        dst_ns = {r["dst_ns"] for r in rs if r["dst_ns"]}
        cross_ns = sum(1 for r in rs if r["src_ns"] and r["dst_ns"] and r["src_ns"] != r["dst_ns"])

        # rolling: tra cuu theo src IP that (da chuan hoa forward) — KHOP key moi
        # trong _rolling_per_source. Tat ca record cung flow_key co cung fwd_src.
        real_src = next(iter(real_src_set)) if real_src_set else normalize_ip(rs[0]["src"])
        roll_ports, roll_ips = roll.get((real_src, win), (0, 0))

        rows.append({
            "flow_key": "|".join(str(x) for x in key),
            "window_start": win,
            "flow_count": n,
            "duration": round(duration, 6),
            "tcp_ratio": round(tcp / n, 4),
            "udp_ratio": round(udp / n, 4),
            "egress_ratio": round(egress / n, 4),
            "is_reply_ratio": round(reply / n, 4),
            "dropped_ratio": round(dropped / n, 4),
            "l7_present_ratio": round(l7 / n, 4),
            "dns_ratio": round(dns / n, 4),
            "distinct_dst_ports": len(dst_ports),
            "distinct_dst_ips": len(dst_ips),
            "distinct_dst_ns": len(dst_ns),
            "cross_ns_ratio": round(cross_ns / n, 4),
            "distinct_dst_ports_60s": roll_ports,
            "distinct_dst_ips_60s": roll_ips,
        })
    return rows


# ----------------------------------------------------------------------------
# NHANH PROCESS (Tetragon)
# ----------------------------------------------------------------------------

# 19 feature: co connect_count (tcp_connect DUOC hook, ~741 event trong log).
# Phai khop KHIT PROCESS_FEATURES trong train.py va fusion.py.
PROCESS_FEATURES = [
    "event_count", "duration",
    "fwd_bytes", "bwd_bytes", "total_bytes",
    "fwd_pkts", "bwd_pkts",
    "bytes_per_sec", "pkts_per_sec",
    "down_up_ratio",
    "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "sendmsg_count", "connect_count", "close_count",
    "distinct_binaries", "uid",
    "distinct_dst_ports_60s", "distinct_dst_ips_60s",
]

FWD_FN = {"tcp_sendmsg"}        # forward bytes (client gui)
BWD_FN = {"tcp_cleanup_rbuf"}   # backward bytes (client nhan)


def load_tetragon(paths):
    recs = []
    skipped = 0
    seen_fns = defaultdict(int)
    has_bwd = False
    for obj in iter_json_lines(paths):
        pk = obj.get("process_kprobe")
        if not pk:
            skipped += 1
            continue
        args = pk.get("args") or []
        sock = None
        nbytes = None
        for a in args:
            if "sock_arg" in a:
                sock = a["sock_arg"]
            elif "int_arg" in a:
                nbytes = a["int_arg"]
        if not sock:
            skipped += 1
            continue
        fn = pk.get("function_name", "")
        seen_fns[fn] += 1
        if fn in BWD_FN:
            has_bwd = True
        try:
            t = parse_time(obj["time"])
        except Exception:
            skipped += 1
            continue
        proc = pk.get("process") or {}
        recs.append({
            "t": t,
            "src": sock.get("saddr", ""),
            "dst": sock.get("daddr", ""),
            "sport": sock.get("sport", 0),
            "dport": sock.get("dport", 0),
            "proto": sock.get("protocol", "").replace("IPPROTO_", ""),
            "fn": fn,
            "bytes": nbytes if nbytes is not None else 0,
            "binary": proc.get("binary", ""),
            "uid": proc.get("uid", -1),
        })
    recs.sort(key=lambda r: r["t"])
    return recs, skipped, dict(seen_fns), has_bwd


def aggregate_process(recs, window, rolling):
    buckets = defaultdict(list)

    # precompute rolling per-source cho nhanh process.
    # Tetragon khong co is_reply -> dung huong forward (src la nguoi khoi tao).
    # Build list (t, dport, dst) theo src IP that cua endpoint forward.
    proc_events = []
    for r in recs:
        if is_junk_endpoint(r["src"], r["proto"]) or is_junk_endpoint(r["dst"], r["proto"]):
            continue
        key, fwd_src, fwd_dst = canonical_flow_key(
            r["src"], r["sport"], r["dst"], r["dport"], r["proto"])
        is_fwd = (normalize_ip(r["src"]), r["sport"]) == fwd_src
        win = int(r["t"] // window) * window
        buckets[(key, win)].append((r, is_fwd))
        if is_fwd:
            # dst that su = fwd_dst; src that su = fwd_src[0]
            proc_events.append({
                "t": r["t"], "src": fwd_src[0],
                "dport": fwd_dst[1], "dst": fwd_dst[0], "is_reply": False,
            })
    roll = _rolling_per_source(proc_events, window, rolling, raw=False)

    rows = []
    for (key, win), items in sorted(buckets.items(), key=lambda kv: kv[0][1]):
        rs = [it[0] for it in items]
        n = len(rs)
        times = [r["t"] for r in rs]
        duration = max(times) - min(times) if n > 1 else 0.0
        # SUA BUG: khong chia cho span do duoc (co the ~0 khi it event) -> rate no
        # len trieu lan. Rate trong cua so W giay phai chia cho it nhat W giay.
        dur_safe = max(duration, float(window))

        fwd_bytes = bwd_bytes = 0
        fwd_pkts = bwd_pkts = 0
        fwd_lens = []
        sendmsg = connect = close = 0
        for r, is_fwd in items:
            b = r["bytes"]
            if r["fn"] in FWD_FN:
                sendmsg += 1
                fwd_bytes += b; fwd_pkts += 1; fwd_lens.append(b)
            elif r["fn"] in BWD_FN:
                bwd_bytes += b; bwd_pkts += 1
            if r["fn"] == "tcp_connect":
                connect += 1
            if r["fn"] == "tcp_close":
                close += 1

        total_bytes = fwd_bytes + bwd_bytes
        total_pkts = fwd_pkts + bwd_pkts
        down_up = (bwd_bytes / fwd_bytes) if fwd_bytes > 0 else 0.0
        binaries = {r["binary"] for r in rs if r["binary"]}
        uid = rs[0]["uid"]

        # rolling: tra cuu tu precompute theo src IP that cua endpoint forward
        fwd_item = next((it for it in items if it[1]), items[0])
        real_src = canonical_flow_key(
            fwd_item[0]["src"], fwd_item[0]["sport"],
            fwd_item[0]["dst"], fwd_item[0]["dport"], fwd_item[0]["proto"])[1][0]
        roll_ports, roll_ips = roll.get((real_src, win), (0, 0))

        rows.append({
            "flow_key": "|".join(str(x) for x in key),
            "window_start": win,
            "event_count": n,
            "duration": round(duration, 6),
            "fwd_bytes": fwd_bytes,
            "bwd_bytes": bwd_bytes,
            "total_bytes": total_bytes,
            "fwd_pkts": fwd_pkts,
            "bwd_pkts": bwd_pkts,
            "bytes_per_sec": round(total_bytes / dur_safe, 3),
            "pkts_per_sec": round(total_pkts / dur_safe, 3),
            "down_up_ratio": round(down_up, 4),
            "fwd_pkt_len_mean": round(sum(fwd_lens) / len(fwd_lens), 3) if fwd_lens else 0.0,
            "fwd_pkt_len_std": round(safe_std(fwd_lens), 3),
            "sendmsg_count": sendmsg,
            "connect_count": connect,
            "close_count": close,
            "distinct_binaries": len(binaries),
            "uid": uid,
            "distinct_dst_ports_60s": roll_ports,
            "distinct_dst_ips_60s": roll_ips,
        })
    return rows


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def write_csv(path, rows, cols):
    key_cols = ["flow_key", "window_start"]
    header = key_cols + cols
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubble", default="/tmp/ebpf-logs/cilium-network.json*")
    ap.add_argument("--tetragon", default="/tmp/ebpf-logs/tetragon-events.json*")
    ap.add_argument("--outdir", default="/tmp/ebpf-logs/features")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rolling", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    hub_paths = expand_paths(args.hubble)
    tet_paths = expand_paths(args.tetragon)
    print(f"[network]  files: {hub_paths}")
    print(f"[process]  files: {tet_paths}")

    # ----- network -----
    h_recs, h_skip = load_hubble(hub_paths)
    n_rows = aggregate_network(h_recs, args.window, args.rolling)
    net_out = os.path.join(args.outdir, "features_network.csv")
    write_csv(net_out, n_rows, NETWORK_FEATURES)
    print(f"[network]  {len(h_recs)} flows -> {len(n_rows)} feature rows "
          f"({h_skip} skipped) -> {net_out}")

    # ----- process -----
    t_recs, t_skip, fns, has_bwd = load_tetragon(tet_paths)
    p_rows = aggregate_process(t_recs, args.window, args.rolling)
    proc_out = os.path.join(args.outdir, "features_process.csv")
    write_csv(proc_out, p_rows, PROCESS_FEATURES)
    print(f"[process]  {len(t_recs)} events -> {len(p_rows)} feature rows "
          f"({t_skip} skipped) -> {proc_out}")
    print(f"[process]  functions seen: {fns}")
    if not has_bwd:
        print("[process]  WARNING: khong thay tcp_cleanup_rbuf -> bwd_bytes = 0. "
              "Bo sung kprobe tcp_cleanup_rbuf trong TracingPolicy neu can byte chieu ve.")

    if len(n_rows) < 50 or len(p_rows) < 50:
        print("\n[!] Du lieu con it. Voi benign baseline can chay capture lau hon "
              "(vai gio) truoc khi train. Day chi la sanity pass.")


if __name__ == "__main__":
    main()
