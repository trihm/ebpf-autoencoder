#!/usr/bin/env python3
"""
Train mot autoencoder cho MOT nhanh (network HOAC process).
Chay hai lan de co dual-AE:

  python3 train.py --branch net  --csv .../features_network.csv --outdir .../models
  python3 train.py --branch proc --csv .../features_process.csv --outdir .../models

Moi lan sinh 3 file khop dung hop dong ma fusion.py mong doi:
  {net|proc}_model.pt    (torch.save ca object model, load bang weights_only=False)
  {net|proc}_scaler.pkl  (StandardScaler da fit tren benign)
  {net|proc}_thr.txt     (1 so: 99.5th percentile reconstruction error benign held-out)

"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- phai trung KHIT voi fusion.py ---
NETWORK_FEATURES = [
    "flow_count", "duration", "tcp_ratio", "udp_ratio",
    "egress_ratio", "is_reply_ratio", "dropped_ratio",
    "l7_present_ratio", "dns_ratio", "distinct_dst_ports",
    "distinct_dst_ips", "distinct_dst_ns", "cross_ns_ratio",
    "distinct_dst_ports_60s", "distinct_dst_ips_60s",
]
PROCESS_FEATURES = [
    "event_count", "duration", "fwd_bytes", "bwd_bytes", "total_bytes",
    "fwd_pkts", "bwd_pkts", "bytes_per_sec", "pkts_per_sec",
    "down_up_ratio", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "sendmsg_count", "connect_count", "close_count",
    "distinct_binaries", "uid",
    "distinct_dst_ports_60s", "distinct_dst_ips_60s",
]

BRANCHES = {
    "net":  ("net",  NETWORK_FEATURES),
    "proc": ("proc", PROCESS_FEATURES),
}

# Cot "rate"/count co duoi dai -> winsorize cap o p99.9 (hoc tu TRAIN split).
# Cap luu ra file, ap LAI Y HET luc eval (fusion.py / evaluate_v3.py) de tranh
# train/eval mismatch. Chi cap tren, khong cap duoi (feature deu >= 0).
WINSOR_COLS = {
    "net": [
        "flow_count", "duration",
        "distinct_dst_ports", "distinct_dst_ips", "distinct_dst_ns",
        # KHONG cap distinct_dst_ports_60s / distinct_dst_ips_60s:
        # hai cot rolling-60s la CHU KY SCAN (nmap no distinct_dst_ports_60s len).
        # Cap chung dong dinh nmap xuong benign -> mat feature phan biet.
        # Duoi dai cua chung LA tin hieu, khong phai nhieu.
    ],
    "proc": [
        "event_count", "duration",
        "fwd_bytes", "bwd_bytes", "total_bytes",
        "fwd_pkts", "bwd_pkts",
        "bytes_per_sec", "pkts_per_sec",
        "down_up_ratio",
        "fwd_pkt_len_mean", "fwd_pkt_len_std",
        "sendmsg_count", "connect_count", "close_count",
        # KHONG cap distinct_dst_ports_60s / distinct_dst_ips_60s (nhu net).
    ],
}


def fit_winsor_caps(X, feature_cols, cols_to_cap, pct):
    """Hoc cap p{pct} cho tung cot trong cols_to_cap, tren X (train split, raw).
    Tra ve numpy array cap[len(feature_cols)]; cot khong cap -> +inf."""
    caps = np.full(len(feature_cols), np.inf, dtype=np.float64)
    idx = {c: i for i, c in enumerate(feature_cols)}
    for c in cols_to_cap:
        if c in idx:
            caps[idx[c]] = float(np.percentile(X[:, idx[c]], pct))
    return caps


def apply_winsor(X, caps):
    """Cap tren tung cot theo caps. X: (n, d) float; caps: (d,)."""
    return np.minimum(X, caps.astype(X.dtype))


class Autoencoder(nn.Module):
    """MLP AE. Bottleneck xuong ~1/3 input roi tai tao. Symmetric encoder/decoder."""
    def __init__(self, n_in):
        super().__init__()
        h1 = max(8, n_in * 2 // 3)      # vd 15 -> 10, 19 -> 12
        h2 = max(4, n_in // 3)          # vd 15 -> 5,  19 -> 6
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(),
            nn.Linear(h1, h2),   nn.ReLU(),
            nn.Linear(h2, h1),   nn.ReLU(),
            nn.Linear(h1, n_in),
        )

    def forward(self, x):
        return self.net(x)


def recon_error(model, X):
    """MSE per-row. X la numpy da scale."""
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(X)).numpy()
    return np.mean((X - recon) ** 2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True, choices=list(BRANCHES.keys()),
                    help="net (Hubble/relation) hoac proc (Tetragon/volume)")
    ap.add_argument("--csv", required=True, help="CSV benign cua nhanh nay")
    ap.add_argument("--outdir", default="/tmp/ebpf-logs/models")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="ty le benign held-out de tinh threshold")
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--winsor-pct", type=float, default=99.9,
                    help="percentile de cap cot rate/count (winsorize). "
                         "Cap hoc tu train split, luu ra {prefix}_winsor.npz.")
    ap.add_argument("--no-winsor", action="store_true",
                    help="tat winsorize (de doi chung truoc/sau).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    prefix, feature_cols = BRANCHES[args.branch]

    # ---- nap du lieu benign ----
    df = pd.read_csv(args.csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{prefix}] CSV thieu cot: {missing}")
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(X_all)
    if n < 50:
        print(f"[{prefix}] CANH BAO: chi {n} dong benign. Threshold se khong on dinh. "
              f"Capture benign lau hon truoc khi tin ket qua.")

    # ---- split train / held-out ----
    idx = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr_raw, X_val_raw = X_all[tr_idx], X_all[val_idx]

    # ---- winsorize: cap cot rate/count o p{winsor_pct}, HOC TU TRAIN split ----
    # Ap TRUOC scaler. Cap luu ra file de eval ap lai y het (tranh mismatch).
    if args.no_winsor:
        caps = np.full(len(feature_cols), np.inf, dtype=np.float64)
        print(f"[{prefix}] winsorize: TAT (--no-winsor)")
    else:
        caps = fit_winsor_caps(X_tr_raw, feature_cols,
                               WINSOR_COLS.get(args.branch, []), args.winsor_pct)
        X_tr_raw = apply_winsor(X_tr_raw, caps)
        X_val_raw = apply_winsor(X_val_raw, caps)
        capped = [(feature_cols[i], caps[i]) for i in range(len(caps))
                  if np.isfinite(caps[i])]
        print(f"[{prefix}] winsorize p{args.winsor_pct} tren {len(capped)} cot: "
              + ", ".join(f"{c}<={v:.3g}" for c, v in capped))

    # ---- scaler fit CHI tren train (da winsorize) ----
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    # ---- train ----
    model = Autoencoder(len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_tr)

    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt))
        epoch_loss = 0.0
        for i in range(0, len(Xt), args.batch):
            b = Xt[perm[i:i + args.batch]]
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[{prefix}] epoch {ep+1:3d}/{args.epochs}  "
                  f"train MSE = {epoch_loss/len(Xt):.6f}")

    # ---- threshold tren held-out benign ----
    val_err = recon_error(model, X_val)
    threshold = float(np.percentile(val_err, args.percentile))

    # ---- luu (khop fusion.py) ----
    model_p  = os.path.join(args.outdir, f"{prefix}_model.pt")
    scaler_p = os.path.join(args.outdir, f"{prefix}_scaler.pkl")
    thr_p    = os.path.join(args.outdir, f"{prefix}_thr.txt")
    winsor_p = os.path.join(args.outdir, f"{prefix}_winsor.npz")

    torch.save(model, model_p)                       # ca object -> weights_only=False
    with open(scaler_p, "wb") as fh:
        pickle.dump(scaler, fh)
    with open(thr_p, "w") as fh:
        fh.write(f"{threshold:.10f}\n")
    # Luu cap + thu tu cot de eval ap lai dung cot (khong phu thuoc vi tri).
    np.savez(winsor_p, caps=caps, feature_cols=np.array(feature_cols, dtype=object))

    frac_flag = float(np.mean(val_err > threshold))
    print(f"[{prefix}] train={len(X_tr)} val={len(X_val)}  "
          f"threshold(p{args.percentile}) = {threshold:.6f}  "
          f"(val flag rate {frac_flag:.3%})")
    print(f"[{prefix}] -> {model_p}")
    print(f"[{prefix}] -> {scaler_p}")
    print(f"[{prefix}] -> {thr_p}")
    print(f"[{prefix}] -> {winsor_p}")


if __name__ == "__main__":
    main()