#!/usr/bin/env python3
"""
Chay ablation kich thuoc bottleneck va bao cao o dung don vi time-bin 5 giay,
dung de lam lai Bang 3.6 (nhanh mang) hoac Bang 3.7 (nhanh tien trinh).

Vi du:
  python3 validate_ablation.py \
      --benign-csv features_network.csv \
      --attack-csv attack_features_network.csv \
      --labels labels.csv \
      --branch net --bottlenecks 3,4,5,6,8 --seeds 42,1,2

Quy trinh moi (bottleneck, seed):
  1. Train autoencoder CHI tren benign, giong train.py (MLP doi xung,
     StandardScaler fit tren train, 100 epoch, Adam lr 1e-3, batch 64).
  2. Nguong = phan vi p99.5 sai so tai tao tren tap validation benign held-out.
  3. Cham diem toan bo file danh gia, gom ve bin 5 giay. Mot bin bi canh bao
     neu CO IT NHAT MOT luong trong bin vuot nguong (dung luat cua do an).
  4. Recall tren cac bin mang nhan tan cong, FPR tren cac bin lanh tinh,
     kem hai cach do do tach biet o muc bin.

Ket qua nhieu seed duoc lay trung binh; script canh bao neu phuong sai giua
cac seed lon (dau hieu ket qua phu thuoc khoi tao chu khong phai kich thuoc
bottleneck).

Luu y: --attack-csv nen la file dac trung cua CA PHIEN danh gia (gom ca cua so
lanh tinh chay nen), khong phai file da loc san chi con tan cong; script tu
gan nhan bang --labels.
"""
import argparse
import sys

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


def recon_error(model, X, batch=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch])
            out.append(np.mean((X[i:i + batch] - model(b).numpy()) ** 2, axis=1))
    return np.concatenate(out) if out else np.zeros(0)


def load_features(path, cols, need_window=False):
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV {path} thieu cot: {missing}")
    if need_window and "window_start" not in df.columns:
        raise SystemExit(f"CSV {path} thieu cot window_start, khong gom bin duoc")
    X = df[cols].to_numpy(dtype=np.float32)
    return df, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def label_rows(df, labels_path):
    """Tra ve mang bool: dong nay nam trong mot khoang tan cong hay khong."""
    lab = pd.read_csv(labels_path)
    start_col = next(c for c in lab.columns if "start" in c.lower())
    end_col = next(c for c in lab.columns if "end" in c.lower())
    starts = pd.to_datetime(lab[start_col], utc=True).astype("int64") // 10 ** 9
    ends = pd.to_datetime(lab[end_col], utc=True).astype("int64") // 10 ** 9
    ws = df["window_start"].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for s, e in zip(starts, ends):
        mask |= (ws >= s) & (ws <= e)
    return mask


def train_one(X_benign, bottleneck, seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.random.permutation(len(X_benign))
    n_val = max(1, int(len(X_benign) * args.val_frac))
    X_val_raw, X_tr_raw = X_benign[idx[:n_val]], X_benign[idx[n_val:]]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    model = Autoencoder(X_benign.shape[1], bottleneck)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_tr)
    model.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch):
            b = Xt[perm[i:i + args.batch]]
            opt.zero_grad()
            loss_fn(model(b), b).backward()
            opt.step()

    err_val = recon_error(model, X_val)
    thr = float(np.percentile(err_val, args.percentile))
    return model, scaler, float(err_val.mean()), thr


def evaluate(model, scaler, thr, X_eval, bins, is_attack_row, bin_size):
    """Gom ve time-bin: bin bi canh bao neu co it nhat mot luong vuot nguong."""
    err = recon_error(model, scaler.transform(X_eval).astype(np.float32))
    bin_id = (bins // bin_size) * bin_size
    d = pd.DataFrame({"bin": bin_id, "err": err, "atk": is_attack_row})
    g = d.groupby("bin").agg(max_err=("err", "max"), atk=("atk", "any"))

    atk = g[g.atk]
    ben = g[~g.atk]
    recall = float((atk.max_err >= thr).mean()) if len(atk) else float("nan")
    fpr = float((ben.max_err >= thr).mean()) if len(ben) else float("nan")
    sep_med = float(np.median(atk.max_err) / max(np.median(ben.max_err), 1e-12)) if len(atk) and len(ben) else float("nan")
    sep_mean = float(atk.max_err.mean() / max(ben.max_err.mean(), 1e-12)) if len(atk) and len(ben) else float("nan")
    return dict(recall=recall, fpr=fpr, sep_med=sep_med, sep_mean=sep_mean,
                n_atk=len(atk), n_ben=len(ben))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-csv", required=True, help="CSV benign dung de train")
    ap.add_argument("--attack-csv", required=True, help="CSV dac trung cua phien danh gia")
    ap.add_argument("--labels", required=True, help="CSV co cot start/end cua tung tan cong")
    ap.add_argument("--branch", default="net", choices=list(BRANCHES))
    ap.add_argument("--bottlenecks", default="3,4,5,6,8")
    ap.add_argument("--seeds", default="42", help="danh sach seed, vd 42,1,2")
    ap.add_argument("--bin-size", type=int, default=5, help="do rong time-bin (giay)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--csv-out", help="ghi ket qua tung seed ra file CSV")
    args = ap.parse_args()

    cols = BRANCHES[args.branch]
    _, X_benign = load_features(args.benign_csv, cols)
    df_eval, X_eval = load_features(args.attack_csv, cols, need_window=True)
    is_attack_row = label_rows(df_eval, args.labels)
    bins = df_eval["window_start"].to_numpy()

    n_atk_bin = len(set((bins[is_attack_row] // args.bin_size) * args.bin_size))
    n_all_bin = len(set((bins // args.bin_size) * args.bin_size))
    print(f"[du lieu] benign train {len(X_benign)} cua so | phien danh gia {len(X_eval)} cua so")
    print(f"[du lieu] tong {n_all_bin} bin {args.bin_size}s, trong do {n_atk_bin} bin co nhan tan cong")
    if n_atk_bin == 0:
        sys.exit("Khong co bin nao mang nhan tan cong — kiem tra lai --labels (mui gio UTC?)")

    sizes = [int(s) for s in args.bottlenecks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for b in sizes:
        for seed in seeds:
            model, scaler, val_loss, thr = train_one(X_benign, b, seed, args)
            r = evaluate(model, scaler, thr, X_eval, bins, is_attack_row, args.bin_size)
            r.update(bottleneck=b, seed=seed, val_loss=val_loss, thr=thr)
            rows.append(r)
            print(f"  b={b:2d} seed={seed:<3d} val_loss={val_loss:.4f} thr={thr:.4f} "
                  f"recall={r['recall']:.3f} fpr={r['fpr']:.3f} sep_med={r['sep_med']:.1f}")

    res = pd.DataFrame(rows)
    if args.csv_out:
        res.to_csv(args.csv_out, index=False)
        print(f"\n[da ghi] {args.csv_out}")

    agg = res.groupby("bottleneck").agg(
        val_loss=("val_loss", "mean"), thr=("thr", "mean"),
        recall=("recall", "mean"), recall_sd=("recall", "std"),
        fpr=("fpr", "mean"), fpr_sd=("fpr", "std"),
        sep_med=("sep_med", "mean"), sep_mean=("sep_mean", "mean")).fillna(0.0)

    print(f"\n=== Trung binh tren {len(seeds)} seed — don vi time-bin {args.bin_size}s ===")
    print(f"{'bottleneck':>10} {'val_loss':>9} {'nguong':>8} {'recall':>8} {'(sd)':>7} "
          f"{'FPR':>7} {'(sd)':>7} {'tach biet':>10}")
    for b, r in agg.iterrows():
        print(f"{b:>10} {r.val_loss:9.4f} {r.thr:8.4f} {r.recall:8.3f} {r.recall_sd:7.3f} "
              f"{r.fpr:7.3f} {r.fpr_sd:7.3f} {r.sep_med:10.1f}")
    print("\n'tach biet' = trung vi sai so lon nhat moi bin tan cong / trung vi cua bin lanh tinh.")

    if len(seeds) > 1:
        noisy = agg[(agg.recall_sd > 0.05) | (agg.fpr_sd > 0.05)]
        if len(noisy):
            print("\nCANH BAO: cac kich thuoc sau co chenh lech giua cac seed > 0,05 "
                  f"({', '.join(str(b) for b in noisy.index)}). "
                  "Ket qua dang phu thuoc khoi tao nhieu hon phu thuoc bottleneck — "
                  "nen tang so seed truoc khi ket luan.")
        else:
            print("\nChenh lech giua cac seed nho (<= 0,05) — ket qua on dinh.")
        best = agg.val_loss.idxmin()
        print(f"val_loss thap nhat: bottleneck {best}. "
              "Nen chon theo val_loss (khai quat tren lanh tinh) de tranh dua thong tin "
              "tu tap tan cong vao buoc chon sieu tham so.")
    else:
        print("\nDang chay 1 seed. Nen chay lai voi --seeds 42,1,2 de kiem tra do on dinh.")


if __name__ == "__main__":
    main()
