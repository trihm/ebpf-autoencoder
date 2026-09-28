#!/usr/bin/env python3
"""
Xac dinh cong thuc "Do tach biet" trong Bang 3.6 cua bao cao.

Cach dung:
  python3 validate_separation.py \
      --benign-csv features_network.csv \
      --attack-csv attack_features_network.csv \
      --labels labels.csv \
      --branch net

Script train lai autoencoder cho tung kich thuoc bottleneck (giong train.py:
cung kien truc MLP doi xung, StandardScaler fit tren train, seed 42, 100 epoch),
roi tinh NHIEU ung vien cong thuc "do tach biet". Cuoi cung so tung ung vien voi
cac gia tri da in trong Bang 3.6 va bao ung vien nao khop nhat.

Neu khong co --attack-csv, script chi in val_loss + threshold (du de kiem tra
Bang 3.7).
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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
BRANCHES = {"net": NETWORK_FEATURES, "proc": PROCESS_FEATURES}

# Gia tri da in trong Bang 3.6 (nhanh mang). Sua lai neu kiem tra bang khac.
REFERENCE = {3: 1216.0, 4: 3255.0, 5: 2312.0, 6: 1736.0, 8: 3210.0}


class Autoencoder(nn.Module):
    def __init__(self, n_in, bottleneck):
        super().__init__()
        h1 = max(8, n_in * 2 // 3)
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(),
            nn.Linear(h1, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, h1), nn.ReLU(),
            nn.Linear(h1, n_in),
        )

    def forward(self, x):
        return self.net(x)


def recon_error(model, X):
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(X)).numpy()
    return np.mean((X - recon) ** 2, axis=1)


def load_features(path, cols):
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV {path} thieu cot: {missing}")
    X = df[cols].to_numpy(dtype=np.float32)
    return df, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def filter_attack(df, X, labels_path):
    """Giu lai cac cua so nam trong khoang thoi gian tan cong."""
    lab = pd.read_csv(labels_path)
    start_col = next(c for c in lab.columns if "start" in c.lower())
    end_col = next(c for c in lab.columns if "end" in c.lower())
    starts = pd.to_datetime(lab[start_col], utc=True).astype("int64") // 10 ** 9
    ends = pd.to_datetime(lab[end_col], utc=True).astype("int64") // 10 ** 9
    ws = df["window_start"].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for s, e in zip(starts, ends):
        mask |= (ws >= s) & (ws <= e)
    print(f"[labels] giu {mask.sum()}/{len(df)} cua so nam trong khoang tan cong")
    return X[mask]


def train_one(X_all, bottleneck, epochs, batch, lr, val_frac, pct, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.random.permutation(len(X_all))
    n_val = max(1, int(len(X_all) * val_frac))
    X_val_raw, X_tr_raw = X_all[idx[:n_val]], X_all[idx[n_val:]]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    model = Autoencoder(X_all.shape[1], bottleneck)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_tr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = Xt[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()

    err_val = recon_error(model, X_val)
    return model, scaler, float(err_val.mean()), float(np.percentile(err_val, pct)), err_val


def candidates(err_val, thr, err_atk):
    """Cac cach dinh nghia 'do tach biet' thuong gap."""
    out = {
        "mean_attack / mean_benign": err_atk.mean() / max(err_val.mean(), 1e-12),
        "median_attack / median_benign": np.median(err_atk) / max(np.median(err_val), 1e-12),
        "max_attack / mean_benign": err_atk.max() / max(err_val.mean(), 1e-12),
        "mean_attack / threshold": err_atk.mean() / max(thr, 1e-12),
        "max_attack / threshold": err_atk.max() / max(thr, 1e-12),
        "p99_attack / threshold": np.percentile(err_atk, 99) / max(thr, 1e-12),
        "z = (mean_a - mean_b) / std_b": (err_atk.mean() - err_val.mean()) / max(err_val.std(), 1e-12),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-csv", required=True)
    ap.add_argument("--attack-csv")
    ap.add_argument("--labels", help="CSV co cot start/end de loc cua so tan cong")
    ap.add_argument("--branch", default="net", choices=list(BRANCHES))
    ap.add_argument("--bottlenecks", default="3,4,5,6,8")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cols = BRANCHES[args.branch]
    _, X_benign = load_features(args.benign_csv, cols)
    X_atk_raw = None
    if args.attack_csv:
        df_a, X_atk_raw = load_features(args.attack_csv, cols)
        if args.labels:
            X_atk_raw = filter_attack(df_a, X_atk_raw, args.labels)

    sizes = [int(s) for s in args.bottlenecks.split(",")]
    results = {}
    for b in sizes:
        model, scaler, val_loss, thr, err_val = train_one(
            X_benign, b, args.epochs, args.batch, args.lr,
            args.val_frac, args.percentile, args.seed)
        line = f"bottleneck {b:2d} | val_loss {val_loss:.4f} | thr(p{args.percentile}) {thr:.4f}"
        if X_atk_raw is not None:
            err_atk = recon_error(model, scaler.transform(X_atk_raw).astype(np.float32))
            results[b] = candidates(err_val, thr, err_atk)
            line += f" | mean_err_attack {err_atk.mean():.4f}"
        print(line)

    if not results:
        return

    print("\n=== Gia tri tung ung vien cong thuc ===")
    names = list(next(iter(results.values())).keys())
    header = "cong thuc".ljust(32) + "".join(f"b={b}".rjust(12) for b in sizes)
    print(header)
    for name in names:
        row = name.ljust(32) + "".join(f"{results[b][name]:12.1f}" for b in sizes)
        print(row)

    ref = {b: v for b, v in REFERENCE.items() if b in results}
    if ref:
        print("\n=== So voi Bang 3.6 ===")
        print("tham chieu".ljust(32) + "".join(f"{ref.get(b, float('nan')):12.1f}" for b in sizes))
        scores = []
        for name in names:
            errs = [abs(results[b][name] - ref[b]) / ref[b] for b in ref]
            scores.append((float(np.mean(errs)), name))
        scores.sort()
        print("\nSai lech tuong doi trung binh (cang nho cang khop):")
        for s, name in scores:
            print(f"  {name:32s} {s * 100:8.1f}%")
        best_err, best = scores[0]
        if best_err < 0.10:
            print(f"\n=> Cong thuc kha nang cao la: {best}")
        else:
            print("\n=> Khong ung vien nao khop duoi 10%. Bang 3.6 co the dung mot dinh nghia khac,\n"
                  "   hoac duoc sinh boi mot lan chay voi seed/du lieu khac. Kiem tra lai script goc.")


if __name__ == "__main__":
    main()
