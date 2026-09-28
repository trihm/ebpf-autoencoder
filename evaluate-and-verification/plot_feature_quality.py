#!/usr/bin/env python3
"""
Ve bieu do CHAT LUONG DAC TRUNG cho dual-AE .

Dung CHUNG logic gan nhan window voi evaluate_v3.py: moi window_start roi vao
[start,end] cua attack nao -> nhan attack do; ngoai tat ca -> benign.

Sinh 4 hinh (300 DPI, font khong dau de tranh loi missing glyph -> dat title
tieng Anh; caption tieng Viet ban tu them trong luan van):

  1. feat_separation_grid.png : histogram log-log tung feature quan trong,
     benign vs attack, ke duong trung vi + p99.5 benign. (khai quat hinh 4.3)
  2. feat_branch_signature.png: voi moi loai attack, feature nao "no" so voi
     benign (ty le median_attack / median_benign) -> cho thay network-branch vs
     process-branch bat attack khac nhau.
  3. feat_correlation.png     : heatmap tuong quan (Spearman) tren benign de lo
     feature du thua -> bien minh kich thuoc bottleneck.
  4. feat_overlap_bar.png     : do tach bach tung feature bang AUC don-bien
     (mot feature phan biet attack/benign tot the nao neu dung rieng).

Chay (giong evaluate_v3.py):
  python3 plot_feature_quality.py \
      --net-csv attack/features/features_network.csv \
      --proc-csv attack/features/features_process.csv \
      --labels labels.csv \
      --benign-net benign/features/features_network.csv \
      --benign-proc benign/features/features_process.csv \
      --window 5 --outdir feat_quality_out

Neu chi co MOT bo CSV (attack co ca benign nen) va khong tach rieng benign,
bo --benign-net/--benign-proc: script se dung window benign trong chinh tap attack.
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

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

# Feature "quan trong" de highlight o hinh 1 (grid histogram). Chon theo lap luan
# separability trong baseline_report: rolling per-source + rate + relation.
HIGHLIGHT = [
    ("distinct_dst_ports_60s", "net"),
    ("distinct_dst_ips_60s",   "net"),
    ("distinct_dst_ports",     "net"),
    ("cross_ns_ratio",         "net"),
    ("pkts_per_sec",           "proc"),
    ("bytes_per_sec",          "proc"),
]


def rfc3339_epoch(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def label_windows(window_starts, labels_df, window):
    """Giong evaluate_v3.py: (y_bin, y_multi)."""
    y_bin = np.zeros(len(window_starts), dtype=int)
    y_multi = np.array(["benign"] * len(window_starts), dtype=object)
    spans = [(r["attack"], rfc3339_epoch(r["start"]), rfc3339_epoch(r["end"]))
             for _, r in labels_df.iterrows()]
    for i, ws in enumerate(window_starts):
        we = ws + window
        for name, s, e in spans:
            if ws < e and we > s:
                y_bin[i] = 1
                y_multi[i] = name
                break
    return y_bin, y_multi


def load_and_label(net_csv, proc_csv, labels_df, window):
    """Merge net+proc theo KEY (outer), gan nhan. Tra ve df day du + list cot."""
    net = pd.read_csv(net_csv)
    proc = pd.read_csv(proc_csv)
    # suffix cot trung ten giua hai nhanh (duration, distinct_dst_*_60s)
    m = net.merge(proc, on=KEY, how="outer", suffixes=("_net", "_proc"))
    y_bin, y_multi = label_windows(m["window_start"].to_numpy(), labels_df, window)
    m["y"] = y_bin
    m["attack"] = y_multi
    return m


def resolve_col(df, name, branch):
    """Sau merge, cot trung ten bi suffix. Tra ve ten cot thuc te trong df."""
    if name in df.columns:
        return name
    cand = f"{name}_{branch}"
    if cand in df.columns:
        return cand
    return None


# ---------------------------------------------------------------------------
# HINH 1: grid histogram benign vs attack cho feature quan trong
# ---------------------------------------------------------------------------
def plot_separation_grid(df, outdir):
    cols = [(resolve_col(df, n, b), n) for n, b in HIGHLIGHT]
    cols = [(c, disp) for c, disp in cols if c is not None]
    ncol = 2
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow))
    axes = np.array(axes).reshape(-1)

    ben = df[df["y"] == 0]
    att = df[df["y"] == 1]

    for ax, (col, disp) in zip(axes, cols):
        b = pd.to_numeric(ben[col], errors="coerce").dropna()
        a = pd.to_numeric(att[col], errors="coerce").dropna()
        b = b[b >= 0]; a = a[a >= 0]
        # log scale: shift 0 -> 0.9 de hien duoc tren truc log
        allv = np.concatenate([b.to_numpy(), a.to_numpy()])
        if allv.size == 0:
            continue
        vmax = max(allv.max(), 1)
        use_log = vmax > 50 and (allv > 0).sum() > 0
        if use_log:
            bpos = np.where(b.to_numpy() <= 0, 0.9, b.to_numpy())
            apos = np.where(a.to_numpy() <= 0, 0.9, a.to_numpy())
            bins = np.logspace(np.log10(0.9), np.log10(vmax * 1.2), 30)
            ax.set_xscale("log")
        else:
            bpos, apos = b.to_numpy(), a.to_numpy()
            bins = np.linspace(0, vmax, 30)

        ax.hist(bpos, bins=bins, alpha=0.6, label=f"benign (n={len(b)})",
                color="steelblue")
        ax.hist(apos, bins=bins, alpha=0.6, label=f"attack (n={len(a)})",
                color="crimson")
        ax.set_yscale("log")
        if len(b) > 0:
            med = np.median(b)
            p995 = np.percentile(b, 99.5)
            ax.axvline(med, color="steelblue", ls="--", lw=1.2,
                       label=f"benign median={med:.0f}")
            ax.axvline(p995, color="darkorange", ls="--", lw=1.2,
                       label=f"benign p99.5={p995:.0f}")
        ax.set_title(disp, fontsize=11)
        ax.set_ylabel("window count (log)")
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Feature separation: benign vs attack (important features)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(outdir, "feat_separation_grid.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# HINH 2: chu ky theo attack — feature nao no manh (log2 fold-change vs benign)
# ---------------------------------------------------------------------------
def plot_branch_signature(df, labels_df, outdir):
    net_cols = [resolve_col(df, c, "net") for c in NETWORK_FEATURES]
    proc_cols = [resolve_col(df, c, "proc") for c in PROCESS_FEATURES]
    feat_pairs = ([(c, n, "network") for c, n in zip(net_cols, NETWORK_FEATURES) if c] +
                  [(c, n, "process") for c, n in zip(proc_cols, PROCESS_FEATURES) if c])

    ben = df[df["y"] == 0]
    ben_med = {}
    for c, n, br in feat_pairs:
        v = pd.to_numeric(ben[c], errors="coerce").dropna()
        ben_med[(n, br)] = np.median(v) if len(v) else 0.0

    attacks = list(labels_df["attack"])
    disp_names = [f"{n}\n({br[:3]})" for c, n, br in feat_pairs]

    mat = np.zeros((len(attacks), len(feat_pairs)))
    for i, atk in enumerate(attacks):
        sub = df[df["attack"] == atk]
        for j, (c, n, br) in enumerate(feat_pairs):
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            am = np.median(v) if len(v) else 0.0
            bm = ben_med[(n, br)]
            eps = 1e-6
            fold = np.log2((am + eps) / (bm + eps))
            mat[i, j] = np.clip(fold, -6, 12)

    # CHIEU DOC: feature o truc Y (doc ngang, khong phai xoay 90), attack o truc X.
    # Chuyen vi ma tran: mat[attack, feature] -> matT[feature, attack].
    matT = mat.T
    fig, ax = plt.subplots(figsize=(0.9 * len(attacks) + 3, 0.32 * len(feat_pairs) + 2))
    im = ax.imshow(matT, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=12)
    ax.set_yticks(range(len(feat_pairs)))
    ax.set_yticklabels(disp_names, fontsize=6.5)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=30, ha="right", fontsize=9)
    # ghi gia tri log2fc vao moi o cho de doc trong ban in
    for fi in range(len(feat_pairs)):
        for ai in range(len(attacks)):
            ax.text(ai, fi, f"{matT[fi, ai]:.0f}", ha="center", va="center",
                    fontsize=5, color="black")
    # duong ngan cach network | process (gio nam NGANG giua hai nhom feature)
    n_net = sum(1 for _, _, br in feat_pairs if br == "network")
    ax.axhline(n_net - 0.5, color="black", lw=1.5)
    # nhan nhom dat ben PHAI, ngoai vung heatmap, de khong de len ten feature
    x_lab = len(attacks) - 0.35
    ax.text(x_lab, n_net / 2, "NETWORK", va="center", ha="left",
            rotation=90, fontsize=9, weight="bold", color="#c0392b")
    ax.text(x_lab, n_net + (len(feat_pairs) - n_net) / 2, "PROCESS",
            va="center", ha="left", rotation=90, fontsize=9, weight="bold",
            color="#2471a3")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log2( median_attack / median_benign )", fontsize=8)
    ax.set_title("Per-attack feature signature (which branch fires)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "feat_branch_signature.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# HINH 3: heatmap tuong quan Spearman tren benign (feature du thua)
# ---------------------------------------------------------------------------
def plot_correlation(df, branch, feature_list, outdir):
    ben = df[df["y"] == 0]
    cols, names = [], []
    for n in feature_list:
        c = resolve_col(df, n, branch)
        if c is None:
            continue
        v = pd.to_numeric(ben[c], errors="coerce")
        if v.std(skipna=True) > 1e-9:
            cols.append(c); names.append(n)
    if len(cols) < 2:
        return None
    sub = ben[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman").to_numpy()

    fig, ax = plt.subplots(figsize=(0.5 * len(names) + 2, 0.5 * len(names) + 2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center",
                    fontsize=5.5, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_title(f"Feature correlation on benign ({branch})", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, f"feat_correlation_{branch}.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# HINH 4: AUC don-bien — moi feature phan biet attack/benign tot the nao neu rieng
# ---------------------------------------------------------------------------
def plot_univariate_auc(df, outdir):
    net_cols = [(resolve_col(df, c, "net"), c, "network") for c in NETWORK_FEATURES]
    proc_cols = [(resolve_col(df, c, "proc"), c, "process") for c in PROCESS_FEATURES]
    pairs = [(c, n, br) for c, n, br in (net_cols + proc_cols) if c is not None]

    y = df["y"].to_numpy()
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None

    rows = []
    for c, n, br in pairs:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
        try:
            auc = roc_auc_score(y, v)
        except ValueError:
            continue
        # AUC < 0.5 nghia la feature nghich dao van phan biet -> lay khoang cach toi 0.5
        sep = abs(auc - 0.5) + 0.5
        rows.append({"feature": n, "branch": br, "auc": auc, "sep": sep})
    if not rows:
        return None
    rdf = pd.DataFrame(rows).sort_values("sep", ascending=True)

    colors = ["#c0392b" if b == "network" else "#2471a3" for b in rdf["branch"]]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rdf) + 1.5))
    ax.barh(range(len(rdf)), rdf["sep"], color=colors)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels([f"{f}" for f in rdf["feature"]], fontsize=8)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Univariate separability (|AUC-0.5|+0.5)")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Per-feature discriminative power (attack vs benign)", fontsize=11)
    # legend thu cong
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b", label="network"),
                       Patch(color="#2471a3", label="process")],
              fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(outdir, "feat_univariate_auc.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    rdf.to_csv(os.path.join(outdir, "feat_univariate_auc.csv"), index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-csv", required=True, help="features_network.csv cua tap ATTACK")
    ap.add_argument("--proc-csv", required=True, help="features_process.csv cua tap ATTACK")
    ap.add_argument("--labels", required=True, help="labels.csv: attack,start,end (RFC3339)")
    ap.add_argument("--benign-net", help="features_network.csv cua tap BENIGN rieng (tuy chon)")
    ap.add_argument("--benign-proc", help="features_process.csv cua tap BENIGN rieng (tuy chon)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--outdir", default="feat_quality_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    labels_df = pd.read_csv(args.labels)

    df = load_and_label(args.net_csv, args.proc_csv, labels_df, args.window)

    # Neu co tap benign rieng: bo benign nen trong tap attack (tranh dem hai lan),
    # chi giu window attack; roi noi benign tu tap rieng vao.
    if args.benign_net and args.benign_proc:
        att_only = df[df["y"] == 1].copy()
        ben_df = load_and_label(args.benign_net, args.benign_proc, labels_df, args.window)
        # tap benign rieng khong chua attack, nhung neu timestamp trung khoang
        # attack thi van bi gan nhan -> ep tat ca ve benign.
        ben_df["y"] = 0
        ben_df["attack"] = "benign"
        # dong cot cho khop (outer-merge co the sinh cot khac nhau giua 2 lan nap)
        all_cols = sorted(set(att_only.columns) | set(ben_df.columns))
        att_only = att_only.reindex(columns=all_cols)
        ben_df = ben_df.reindex(columns=all_cols)
        df = pd.concat([ben_df, att_only], ignore_index=True)
        print(f"[benign] dung tap benign rieng: {len(ben_df)} window benign")

    n_att = int(df["y"].sum()); n_ben = len(df) - n_att
    print(f"[label] {len(df)} windows: {n_att} attack, {n_ben} benign")
    if n_att == 0:
        raise SystemExit("[!] Khong window attack nao. Kiem tra timezone giua log va labels.csv.")

    outs = []
    outs.append(plot_separation_grid(df, args.outdir))
    outs.append(plot_branch_signature(df, labels_df, args.outdir))
    outs.append(plot_correlation(df, "net", NETWORK_FEATURES, args.outdir))
    outs.append(plot_correlation(df, "proc", PROCESS_FEATURES, args.outdir))
    outs.append(plot_univariate_auc(df, args.outdir))

    print("\n[out] figures:")
    for p in outs:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()