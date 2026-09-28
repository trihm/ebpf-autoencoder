#!/usr/bin/env python3
"""
Danh gia dual-AE tren du lieu attack co nhan thoi gian.

Vao:
  - features_network.csv + features_process.csv  (da aggregate tu log ATTACK)
  - 6 file model (net/proc x model/scaler/thr)   (train tren BENIGN)
  - labels.csv: attack,start,end (RFC3339)        (khoang thoi gian moi attack)

Lam:
  1. Cham reconstruction error tung nhanh, fuse (max) -> score lien tuc + alert.
  2. Gan nhan window: window_start roi vao [start,end] cua attack nao -> nhan attack do;
     ngoai tat ca -> benign (0).
  3. Metric:
     - ROC-AUC toan cuc (score lien tuc vs nhan nhi phan benign/attack)
     - Precision / Recall / F1 toan cuc tai threshold (alert)
     - P/R/F1 tach RIENG tung attack (recall theo loai: nmap/hping3/slowloris/brute)
     - So sanh Isolation Forest baseline (train lai tren benign network+proc ghep)
  4. Xuat figure 300 DPI: ROC curve, error timeline voi vung attack to mau.

Chay:
  python3 evaluate.py \
      --net-csv attack/features/features_network.csv \
      --proc-csv attack/features/features_process.csv \
      --net-model models/net_model.pt --net-scaler models/net_scaler.pkl --net-thr models/net_thr.txt \
      --proc-model models/proc_model.pt --proc-scaler models/proc_scaler.pkl --proc-thr models/proc_thr.txt \
      --labels labels.csv \
      --benign-net benign/features/features_network.csv \
      --benign-proc benign/features/features_process.csv \
      --outdir eval_out
"""
import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

KEY = ["flow_key", "window_start"]

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


# phai KHOP train.py
class Autoencoder(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        h1 = max(8, n_in * 2 // 3)
        h2 = max(4, n_in // 3)
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(),
            nn.Linear(h1, h2),   nn.ReLU(),
            nn.Linear(h2, h1),   nn.ReLU(),
            nn.Linear(h1, n_in),
        )

    def forward(self, x):
        return self.net(x)


def load_winsor_caps(scaler_p, cols):
    """Tim {prefix}_winsor.npz canh scaler. Tra ve caps[len(cols)] theo dung
    THU TU cols cua eval (map bang ten, khong theo vi tri). Khong co file ->
    None (tuong thich nguoc voi model train truoc khi co winsorize)."""
    winsor_p = scaler_p.replace("_scaler.pkl", "_winsor.npz")
    if not os.path.exists(winsor_p):
        return None
    d = np.load(winsor_p, allow_pickle=True)
    saved_caps = d["caps"]
    saved_cols = list(d["feature_cols"])
    idx = {c: i for i, c in enumerate(saved_cols)}
    caps = np.full(len(cols), np.inf, dtype=np.float64)
    for j, c in enumerate(cols):
        if c in idx:
            caps[j] = saved_caps[idx[c]]
    return caps


def recon_error(model, scaler, df, cols, caps=None):
    X = df[cols].to_numpy(dtype=np.float32)
    if caps is not None:
        X = np.minimum(X, caps.astype(X.dtype))   # winsorize TRUOC scaler, y het train
    X = scaler.transform(X)
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(X)).numpy()
    return np.mean((X - recon) ** 2, axis=1)


def load_branch(csv, model_p, scaler_p, thr_p, cols):
    df = pd.read_csv(csv)
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"[eval] {csv} thieu cot: {miss}")
    model = torch.load(model_p, map_location="cpu", weights_only=False)
    with open(scaler_p, "rb") as fh:
        scaler = pickle.load(fh)
    with open(thr_p) as fh:
        thr = float(fh.read().strip())
    caps = load_winsor_caps(scaler_p, cols)
    if caps is not None:
        n_cap = int(np.isfinite(caps).sum())
        print(f"[eval] winsorize: ap {n_cap} cot tu {scaler_p.replace('_scaler.pkl','_winsor.npz')}")
    else:
        print(f"[eval] winsorize: khong tim thay file cap -> bo qua (model cu)")
    df = df.copy()
    df["err"] = recon_error(model, scaler, df, cols, caps=caps)
    return df[KEY + ["err"]], thr


def rfc3339_epoch(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def label_windows(window_starts, labels_df, window):
    """Tra ve (y_bin, y_multi). Window [ws, ws+window) giao voi [start,end] -> attack."""
    y_bin = np.zeros(len(window_starts), dtype=int)
    y_multi = np.array(["benign"] * len(window_starts), dtype=object)
    spans = [(r["attack"], rfc3339_epoch(r["start"]), rfc3339_epoch(r["end"]))
             for _, r in labels_df.iterrows()]
    for i, ws in enumerate(window_starts):
        we = ws + window
        for name, s, e in spans:
            if ws < e and we > s:      # giao nhau
                y_bin[i] = 1
                y_multi[i] = name
                break
    return y_bin, y_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-csv", required=True)
    ap.add_argument("--proc-csv", required=True)
    ap.add_argument("--net-model", required=True)
    ap.add_argument("--net-scaler", required=True)
    ap.add_argument("--net-thr", required=True)
    ap.add_argument("--proc-model", required=True)
    ap.add_argument("--proc-scaler", required=True)
    ap.add_argument("--proc-thr", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--benign-net", help="de train Isolation Forest baseline")
    ap.add_argument("--benign-proc", help="de train Isolation Forest baseline")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- cham dual-AE ----
    net_df, net_thr = load_branch(args.net_csv, args.net_model, args.net_scaler,
                                  args.net_thr, NETWORK_FEATURES)
    proc_df, proc_thr = load_branch(args.proc_csv, args.proc_model, args.proc_scaler,
                                    args.proc_thr, PROCESS_FEATURES)

    m = net_df.merge(proc_df, on=KEY, how="outer", suffixes=("_net", "_proc"))
    m["norm_net"] = m["err_net"] / net_thr
    m["norm_proc"] = m["err_proc"] / proc_thr
    m["score"] = m[["norm_net", "norm_proc"]].max(axis=1, skipna=True)
    m["alert"] = m["score"] >= 1.0

    # ---- gan nhan ----
    labels_df = pd.read_csv(args.labels)
    ws = m["window_start"].to_numpy()
    y_bin, y_multi = label_windows(ws, labels_df, args.window)
    m["y"] = y_bin
    m["attack"] = y_multi

    n_att = int(y_bin.sum())
    n_ben = len(y_bin) - n_att
    print(f"[label] {len(m)} windows: {n_att} attack, {n_ben} benign")
    if n_att == 0:
        raise SystemExit("[!] Khong window nao roi vao khoang attack. "
                         "Kiem tra timezone/timestamp giua log va labels.csv.")

    # ---- metric toan cuc (dual-AE) ----
    score = m["score"].to_numpy()
    pred = m["alert"].to_numpy().astype(int)
    auc = roc_auc_score(y_bin, score)
    P = precision_score(y_bin, pred, zero_division=0)
    R = recall_score(y_bin, pred, zero_division=0)
    F1 = f1_score(y_bin, pred, zero_division=0)
    fpr_val = pred[y_bin == 0].mean() if n_ben else float("nan")

    print("\n=== DUAL-AE (fusion max) ===")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"Precision      : {P:.4f}")
    print(f"Recall         : {R:.4f}")
    print(f"F1             : {F1:.4f}")
    print(f"False Pos Rate : {fpr_val:.4f}")

    # ---- Recall per-attack: CA HAI don vi (per-flow VA time-bin) ----
    # Precision/F1/FPR chi co nghia o muc tong (mot false positive khong thuoc
    # attack nao) -> per-attack chi bao RECALL. P/F1/FPR tong o khoi rieng ben duoi.
    print("\n=== Recall per-attack (per-flow) ===")
    # tinh time-bin recall per-attack: can bang time-bin da gan nhan (lam ben duoi),
    # nen tinh o day bang cach gom truoc.
    tb_pre = m.groupby("window_start").agg(
        score=("score", "max"),
    ).reset_index()
    tb_pre["alert"] = tb_pre["score"] >= 1.0
    attack_of_bin_pre = (m[m["attack"] != "benign"]
                         .groupby("window_start")["attack"]
                         .agg(lambda s: s.value_counts().index[0]))
    tb_pre["attack"] = tb_pre["window_start"].map(attack_of_bin_pre).fillna("benign")

    per_rows = []
    for name in labels_df["attack"]:
        mask = m["attack"] == name
        tot = int(mask.sum())
        hit = int(m.loc[mask, "alert"].sum())
        rec_flow = hit / tot if tot else float("nan")
        # time-bin recall cho attack nay
        tbm = tb_pre["attack"] == name
        tb_tot = int(tbm.sum())
        tb_hit = int(tb_pre.loc[tbm, "alert"].sum())
        rec_bin = tb_hit / tb_tot if tb_tot else float("nan")
        sub = m[mask]
        net_hits = int((sub["norm_net"] >= 1.0).sum())
        proc_hits = int((sub["norm_proc"] >= 1.0).sum())
        dominant = "network" if net_hits > proc_hits else "process"
        per_rows.append({
            "attack": name,
            "flow_windows": tot, "flow_detected": hit,
            "recall_flow": round(rec_flow, 4),
            "time_bins": tb_tot, "bins_detected": tb_hit,
            "recall_timebin": round(rec_bin, 4),
            "dominant_branch": dominant,
        })
        print(f"  {name:18s} per-flow={rec_flow:.3f} ({hit}/{tot})  "
              f"time-bin={rec_bin:.3f} ({tb_hit}/{tb_tot})  nhanh: {dominant}")
    per_df = pd.DataFrame(per_rows)

    # ==== DANH GIA THEO TIME-BIN (chuan IDS, sua bug don vi flow) ====
    # Gom moi flow cung window_start thanh 1 time-bin. Bin do "alert" neu BAT KY
    # flow nao trong bin vuot threshold. Tranh viec 1 cuoc tan cong (vd hping3 flood
    # = hang tram nghin flow SYN, hay slowloris = hang nghin ket noi im lang) bi
    # dem thanh hang tram nghin diem lam loang metric.
    print("\n=== DANH GIA THEO TIME-BIN (5s) ===")
    tb = m.groupby("window_start").agg(
        score=("score", "max"),
        y=("y", "max"),
    ).reset_index()
    # nhan multi-class cho time-bin: attack chiem da so trong bin
    attack_of_bin = (m[m["attack"] != "benign"]
                     .groupby("window_start")["attack"]
                     .agg(lambda s: s.value_counts().index[0]))
    tb["attack"] = tb["window_start"].map(attack_of_bin).fillna("benign")
    tb["alert"] = tb["score"] >= 1.0

    tb_y = tb["y"].to_numpy()
    tb_pred = tb["alert"].to_numpy().astype(int)
    tb_score = tb["score"].to_numpy()
    n_att_bin = int(tb_y.sum()); n_ben_bin = len(tb_y) - n_att_bin
    tb_auc = None   # de figure biet co ve duong time-bin hay khong
    if n_att_bin and n_ben_bin:
        tb_auc = roc_auc_score(tb_y, tb_score)
        tb_P = precision_score(tb_y, tb_pred, zero_division=0)
        tb_R = recall_score(tb_y, tb_pred, zero_division=0)
        tb_F1 = f1_score(tb_y, tb_pred, zero_division=0)
        tb_fpr = tb_pred[tb_y == 0].mean()
        print(f"time-bins: {len(tb)} ({n_att_bin} attack, {n_ben_bin} benign)")
        print(f"ROC-AUC  : {tb_auc:.4f}")
        print(f"Precision: {tb_P:.4f}   Recall: {tb_R:.4f}   F1: {tb_F1:.4f}")
        print(f"FPR      : {tb_fpr:.4f}")

    # ==== PER-INTERVAL DETECTION + LATENCY ====
    # Moi attack = 1 su kien. "Phat hien" neu co >=1 time-bin alert trong khoang.
    # Latency = tu attack start den time-bin alert dau tien.
    print("\n=== PHAT HIEN THEO SU KIEN + DO TRE ===")
    inter_rows = []
    for _, r in labels_df.iterrows():
        name = r["attack"]
        s = rfc3339_epoch(r["start"]); e = rfc3339_epoch(r["end"])
        bins_in = tb[(tb["window_start"] >= s - args.window) &
                     (tb["window_start"] < e)]
        n_bins = len(bins_in)
        alert_bins = bins_in[bins_in["alert"]]
        detected = len(alert_bins) > 0
        bin_recall = len(alert_bins) / n_bins if n_bins else float("nan")
        # latency: time-bin alert dau tien - start
        if detected:
            first = alert_bins["window_start"].min()
            latency = max(0.0, first - s)
        else:
            latency = float("nan")
        inter_rows.append({
            "attack": name, "detected": detected,
            "bins_total": n_bins, "bins_alerted": len(alert_bins),
            "bin_recall": round(bin_recall, 3) if n_bins else None,
            "latency_s": round(latency, 1) if detected else None,
        })
        status = "PHAT HIEN" if detected else "BO SOT"
        lat = f"{latency:.0f}s" if detected else "-"
        print(f"  {name:18s} {status:10s}  bins {len(alert_bins)}/{n_bins}  "
              f"recall={bin_recall:.2f}  latency={lat}")
    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_csv(os.path.join(args.outdir, "per_interval.csv"), index=False)

    # ---- Isolation Forest baseline ----
    if args.benign_net and args.benign_proc:
        print("\n=== Isolation Forest baseline ===")
        # ghep feature network+proc tren benign de fit, roi cham tren attack
        bn = pd.read_csv(args.benign_net)
        bp = pd.read_csv(args.benign_proc)
        bmerge = bn.merge(bp, on=KEY, suffixes=("_net", "_proc"))
        feat_net = [f"{c}" for c in NETWORK_FEATURES]
        feat_proc = [f"{c}" for c in PROCESS_FEATURES]
        # sau merge, ten trung (duration, distinct_dst_ports_60s...) bi suffix
        # -> lay tat ca cot so tru khoa
        num_cols = [c for c in bmerge.columns if c not in KEY]
        bmerge_num = bmerge[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        # loc cot chet
        keep = [c for c in num_cols if bmerge_num[c].std() > 1e-9]
        Xb = bmerge_num[keep].to_numpy()
        sc = StandardScaler().fit(Xb)
        iso = IsolationForest(n_estimators=200, contamination=0.005,
                              random_state=42).fit(sc.transform(Xb))

        # cham tren attack: ghep tuong tu
        an = pd.read_csv(args.net_csv)
        apd = pd.read_csv(args.proc_csv)
        amerge = an.merge(apd, on=KEY, suffixes=("_net", "_proc"))
        amerge_num = amerge[keep].apply(pd.to_numeric, errors="coerce").fillna(0)
        iso_raw = -iso.score_samples(sc.transform(amerge_num.to_numpy()))  # cao = bat thuong
        iso_pred = (iso.predict(sc.transform(amerge_num.to_numpy())) == -1).astype(int)

        # gan nhan cho amerge
        yb_iso, _ = label_windows(amerge["window_start"].to_numpy(), labels_df, args.window)
        if yb_iso.sum() > 0 and (yb_iso == 0).sum() > 0:
            iso_auc = roc_auc_score(yb_iso, iso_raw)
            iso_P = precision_score(yb_iso, iso_pred, zero_division=0)
            iso_R = recall_score(yb_iso, iso_pred, zero_division=0)
            iso_F1 = f1_score(yb_iso, iso_pred, zero_division=0)
            print(f"ROC-AUC : {iso_auc:.4f}   P: {iso_P:.4f}  R: {iso_R:.4f}  F1: {iso_F1:.4f}")
            print(f"(chi tren {len(amerge)} window co CA hai nhanh — inner join)")
        else:
            print("[!] khong du window inner-join co nhan de danh gia baseline.")
            iso_auc = iso_raw  # placeholder khong dung

    # ---- xuat CSV ket qua ----
    m.sort_values("score", ascending=False).to_csv(
        os.path.join(args.outdir, "scored_windows.csv"), index=False)
    per_df.to_csv(os.path.join(args.outdir, "per_attack.csv"), index=False)

    # ---- figure 1: ROC curve ----
    # Ve CA HAI duong: time-bin (chuan IDS, con so de bao cao) va per-flow
    # (bi loang boi flood, chi de tham khao). Duong time-bin la net dam.
    plt.figure(figsize=(5, 5))
    if tb_auc is not None:
        fpr_tb, tpr_tb, _ = roc_curve(tb_y, tb_score)
        plt.plot(fpr_tb, tpr_tb, lw=2.2, color="C0",
                 label=f"Time-bin (AUC={tb_auc:.3f})")
    fpr, tpr, _ = roc_curve(y_bin, score)
    plt.plot(fpr, tpr, lw=1.3, ls="--", color="C1", alpha=0.8,
             label=f"Per-flow (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Dual Autoencoder")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roc_curve.png"), dpi=300)
    plt.close()

    # roc rieng time-bin (dung cho thesis figure)
    if tb_auc is not None:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr_tb, tpr_tb, lw=2.2, color="C0",
                 label=f"Dual-AE time-bin (AUC={tb_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC — Dual Autoencoder (time-bin)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "roc_curve_timebin.png"), dpi=300)
        plt.close()

    # ---- figure 2: error timeline voi vung attack ----
    plt.figure(figsize=(11, 4))
    t0 = ws.min()
    rel = (ws - t0)
    plt.scatter(rel[y_bin == 0], score[y_bin == 0], s=6, alpha=0.4,
                color="steelblue", label="benign")
    plt.scatter(rel[y_bin == 1], score[y_bin == 1], s=6, alpha=0.6,
                color="crimson", label="attack window")
    plt.axhline(1.0, color="black", ls="--", lw=1, label="threshold")
    # to bong vung attack
    for _, r in labels_df.iterrows():
        s = rfc3339_epoch(r["start"]) - t0
        e = rfc3339_epoch(r["end"]) - t0
        plt.axvspan(s, e, alpha=0.12, color="red")
        plt.text((s + e) / 2, plt.ylim()[1] * 0.92, r["attack"],
                 ha="center", fontsize=7, rotation=0)
    plt.yscale("log")
    plt.xlabel("Time since start (s)")
    plt.ylabel("Fusion score (log)")
    plt.title("Reconstruction-error timeline")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "error_timeline.png"), dpi=300)
    plt.close()

    print(f"\n[out] -> {args.outdir}/scored_windows.csv")
    print(f"[out] -> {args.outdir}/per_attack.csv")
    print(f"[out] -> {args.outdir}/roc_curve.png")
    print(f"[out] -> {args.outdir}/error_timeline.png")


if __name__ == "__main__":
    main()