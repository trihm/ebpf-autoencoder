#!/usr/bin/env python3
"""
extra_experiments.py — cac thi nghiem bo sung .

Dat CUNG THU MUC voi train.py va rq_experiments.py (script import lai cac ham cua hai file do).

  E1  Kiem tra lai RQ2/RQ3 (khong huan luyen lai, dung model da luu):
        - recall theo tung tan cong o FPR khung 0,05 va 0,089 cho M1/M2/M3 (kem so dem, Wilson)
        - H3: FPR khung du doan E[1-(1-a)^N] tren phan phoi N thuc do (a do duoc va a muc tieu)
        - Cung phan vi nguong cho hai goc nhin (99,5/99,5 va 99,9/99,9)
  E2  Nhieu seed + bootstrap theo khoi co ghep cap (C3):
        - huan luyen lai net/proc/M4 voi cac seed (mac dinh 42,1,2,3,4)
        - ROC-AUC va recall tai cung FPR cua M1..M4: trung binh +- do lech chuan
        - bootstrap khoi truot (moving block) co ghep cap cho chenh lech AUC M1-M4, M1-M3, M1-M2
        - KTC bootstrap khoi cho AUC, recall, FPR cua M1 (so voi Wilson/bootstrap i.i.d.)
  E3  Bo dac trung danh tinh (C2): huan luyen lai goc nhin tien trinh va M4 KHONG co uid, distinct_binaries
  E4  Tiem bat thuong co kiem soat (C1): cong delta (don vi do lech chuan) vao k dac trung cua MOT goc nhin
        tren cua so-luong lanh tinh co du hai goc nhin; do ty le phat hien cua M1, M3, M4 o cung FPR
        -> duong phat hien theo delta, delta50 va kappa thuc nghiem.

Vi du:
  python3 extra_experiments.py \\
      --benign-net  benign/features/features_network.csv \\
      --benign-proc benign/features/features_process.csv \\
      --attack-net  attack/features/features_network.csv \\
      --attack-proc attack/features/features_process.csv \\
      --labels labels.csv --models run/models \\
      --attacker-ip 10.0.0.20,10.0.0.152 --net-pct 99.9 --proc-pct 99.5 \\
      --outdir extra_out

Dau ra (--outdir): extra_results.md (gui lai file nay), extra_results.json, fig_injection.png.
Thoi gian: ~10–30 phut tren CPU (khoang 17 lan huan luyen AE). Chay tung phan: --only E1,E4 ...
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq_experiments as Q  # noqa: E402  (Q.T la module train.py)
import __main__  # noqa: E402
__main__.Autoencoder = Q.T.Autoencoder

T = Q.T
NET, PROC, KEY, ATTACKS = Q.NET, Q.PROC, Q.KEY, Q.ATTACKS
IDENTITY = ["uid", "distinct_binaries"]
OUT = {}      # ket qua dang cay (json)
MD = []       # cac dong Markdown


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def f3(x):
    return Q.f3(float(x)) if x is not None else "—"


def f2(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}".replace(".", ",")


def ci(lo, hi, prop=True):
    """KTC; prop=True: ty le (cat ve [0; 1]); prop=False: chenh lech (giu gia tri am)."""
    if prop:
        lo = max(0.0, lo) if lo == lo else lo
        hi = min(1.0, hi) if hi == hi else hi
    return f"[{f3(lo)}; {f3(hi)}]"


def md_table(header, rows):
    MD.append("| " + " | ".join(header) + " |")
    MD.append("|" + "---|" * len(header))
    for r in rows:
        MD.append("| " + " | ".join(str(x) for x in r) + " |")
    MD.append("")


# =========================================================================== mo hinh tong quat
def train_view_cols(df, cols, prefix, pctl, seed, a):
    """Nhu train.py nhung voi tap cot tuy chon. Tra ve (ViewModel, sai so tren phan giu lai)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = df[cols].to_numpy(np.float32)
    n = len(X)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * 0.2))
    va, tr = idx[:n_val], idx[n_val:]
    wcols = [c for c in T.WINSOR_COLS[prefix] if c in cols]
    caps = T.fit_winsor_caps(X[tr], cols, wcols, 99.9)
    Xtr, Xva = T.apply_winsor(X[tr], caps), T.apply_winsor(X[va], caps)
    sc = StandardScaler().fit(Xtr)
    model = Q.fit_ae(sc.transform(Xtr).astype(np.float32), seed, a.epochs, a.batch, a.lr,
                     f"{prefix}/{len(cols)}c/seed{seed}")
    err = Q.sq_err_model(model, sc.transform(Xva).astype(np.float32)).mean(1)
    return Q.ViewModel(model, sc, caps, float(np.percentile(err, pctl)), cols), err


class JP:
    """Bieu dien ghep [net, proc, co_khuyet_net, co_khuyet_proc] voi tap cot tuy chon."""

    def __init__(self, cols):
        self.cols = cols  # {"net": [...], "proc": [...]}

    def fit(self, J, tr):
        self.caps, self.sc = {}, {}
        for v in ("net", "proc"):
            c = [f"{v}__{x}" for x in self.cols[v]]
            pres = J[f"has_{v}"].to_numpy()
            X = J.loc[tr[pres[tr]], c].to_numpy(np.float32)
            wc = [x for x in T.WINSOR_COLS[v] if x in self.cols[v]]
            self.caps[v] = T.fit_winsor_caps(X, self.cols[v], wc, 99.9)
            self.sc[v] = StandardScaler().fit(T.apply_winsor(X, self.caps[v]))
        return self

    def transform(self, J):
        parts = []
        for v in ("net", "proc"):
            c = [f"{v}__{x}" for x in self.cols[v]]
            pres = J[f"has_{v}"].to_numpy()
            Z = np.zeros((len(J), len(c)), np.float32)
            if pres.any():
                X = np.nan_to_num(J.loc[pres, c].to_numpy(np.float32))
                Z[pres] = self.sc[v].transform(T.apply_winsor(X, self.caps[v]))
            parts.append(Z)
        fl = np.stack([~J["has_net"].to_numpy(), ~J["has_proc"].to_numpy()], 1).astype(np.float32)
        return np.concatenate(parts + [fl], 1)


def train_joint(Jb, cols, seed, pctl, a):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.random.permutation(len(Jb))
    n_val = max(1, int(len(Jb) * 0.2))
    va, tr = idx[:n_val], idx[n_val:]
    prep = JP(cols).fit(Jb, tr)
    Xb = prep.transform(Jb)
    m = Q.fit_ae(Xb[tr], seed, a.epochs, a.batch, a.lr, f"M4/{Xb.shape[1]}d/seed{seed}")
    e_va = Q.sq_err_model(m, Xb[va]).mean(1)
    return {"model": m, "prep": prep, "tau": float(np.percentile(e_va, pctl)), "e_va": e_va}


def score(J, vm, m4=None):
    """Diem cua so-luong cho M1..M4 (khong ghi vao J). vm: {'net': ViewModel, 'proc': ViewModel}."""
    e = {}
    for v in ("net", "proc"):
        pres = J[f"has_{v}"].to_numpy()
        arr = np.full(len(J), np.nan)
        if pres.any():
            X = J.loc[pres, [f"{v}__{c}" for c in vm[v].cols]].to_numpy(np.float64)
            arr[pres] = vm[v].sq_errors(X).mean(1)
        e[v] = arr
    zn, zp = e["net"] / vm["net"].thr, e["proc"] / vm["proc"].thr
    Z = np.stack([zn, zp], 1)
    dn, dp = len(vm["net"].cols), len(vm["proc"].cols)
    hn, hp = ~np.isnan(zn), ~np.isnan(zp)
    s = {"net": zn, "proc": zp,
         "M1": np.nanmax(np.where(np.isnan(Z), -np.inf, Z), 1),
         "M2": np.nanmean(Z, 1),
         "M3": (dn * np.nan_to_num(e["net"]) + dp * np.nan_to_num(e["proc"])) / (dn * hn + dp * hp)}
    s["M1"][~(hn | hp)] = np.nan
    if m4 is not None:
        s["M4"] = Q.sq_err_model(m4["model"], m4["prep"].transform(J)).mean(1)
    return s


def to_bin(ws, sc, bins_ws):
    b = pd.Series(sc).groupby(ws).max()
    return b.reindex(bins_ws).fillna(0.0).to_numpy()


def thr_fpr(s_benign, target):
    return Q.thr_at_fpr(s_benign, target)


def recall_block(al, att):
    out = {}
    for k in ATTACKS:
        m = att == k
        out[k] = (int(al[m].sum()), int(m.sum()))
    return out


def rec_str(kn):
    k, n = kn
    if n == 0:
        return "—"
    lo, hi = Q.wilson(k, n)
    return f"{f3(k / n)} ({k}/{n}) {ci(lo, hi)}"


def block_idx(n, L, rng):
    starts = rng.integers(0, max(1, n - L + 1), int(math.ceil(n / L)))
    return np.concatenate([np.arange(s, min(s + L, n)) for s in starts])[:n]


def boot(y, fn, n_boot, seed, L=None):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        i = block_idx(n, L, rng) if L else rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(i))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("benign-net", "benign-proc", "attack-net", "attack-proc", "labels", "models"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--attacker-ip", default="")
    ap.add_argument("--outdir", default="extra_out")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--net-pct", type=float, default=99.9)
    ap.add_argument("--proc-pct", type=float, default=99.5)
    ap.add_argument("--joint-pct", type=float, default=99.5)
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--block", type=int, default=12, help="do dai khoi (so khung) cho bootstrap khoi; 12 = 1 phut")
    ap.add_argument("--inj-base", type=int, default=5000, help="so cua so-luong lanh tinh dung de tiem")
    ap.add_argument("--inj-k", default="1,2,4")
    ap.add_argument("--inj-delta", default="0,0.5,1,1.5,2,3,4,6,8,12,16")
    ap.add_argument("--inj-fpr", default="0.01,0.005", help="FPR muc cua so-luong de dat nguong khi tiem")
    ap.add_argument("--only", default="E1,E2,E3,E4",
                    help="cac phan can chay; E3S = bo uid lap lai voi nhieu seed (--e3-seeds)")
    ap.add_argument("--e3-seeds", default="42,1,2,3,4", help="cac seed cho E3S")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    only = set(a.only.split(","))
    W = a.window
    seeds = [int(x) for x in a.seeds.split(",")]
    if 42 not in seeds:
        seeds = [42] + seeds

    # ------------------------------------------------------------ du lieu (nhu rq_experiments.py)
    log("doc du lieu")
    bn, bp = Q.read_csv(a.benign_net, NET), Q.read_csv(a.benign_proc, PROC)
    an, apd = Q.read_csv(a.attack_net, NET), Q.read_csv(a.attack_proc, PROC)
    lab = pd.read_csv(a.labels)
    spans = [(Q.canon(r["attack"]), Q.rfc3339(r["start"]), Q.rfc3339(r["end"])) for _, r in lab.iterrows()]
    vm = {"net": Q.ViewModel.load(a.models, "net"), "proc": Q.ViewModel.load(a.models, "proc")}
    pct = {"net": a.net_pct, "proc": a.proc_pct}
    val_err = {}
    for v, df in (("net", bn), ("proc", bp)):
        vi = Q.random_val_idx(len(df), 42)
        val_err[v] = vm[v].sq_errors(df[Q.COLS[v]].to_numpy(np.float64)[vi]).mean(1)
        rank = 100 * float(np.mean(val_err[v] <= vm[v].thr))
        if abs(rank - pct[v]) > 0.3:
            raise SystemExit(f"[DUNG] model {v}: nguong o phan vi {rank:.2f}, khac --{v}-pct {pct[v]}. "
                             f"Kiem tra lai --models / CSV lanh tinh.")
    J = Q.join_views(an, apd)
    Jb = Q.join_views(bn, bp)
    ws = J["window_start"].to_numpy()
    att_flow = Q.interval_labels(ws, spans, W)
    bins_ws = np.sort(np.unique(ws))
    att = Q.interval_labels(bins_ws, spans, W)
    y = (att != "benign").astype(int)
    nb = len(bins_ws)
    log(f"{len(J)} cua so-luong, {nb} khung ({int(y.sum())} tan cong)")
    MD.append("# Ket qua thi nghiem bo sung (v13)\n")
    MD.append(f"Phien danh gia: {nb} khung, {int(y.sum())} tan cong, {int((y == 0).sum())} lanh tinh. "
              f"Seed: {seeds}. Bootstrap: {a.n_boot} lan, khoi {a.block} khung.\n")

    s42 = score(J, vm)
    B42 = {k: to_bin(ws, s42[k], bins_ws) for k in ("M1", "M2", "M3", "net", "proc")}
    al_M1 = to_bin(ws, (s42["M1"] >= 1).astype(float), bins_ws) > 0
    fpr_M1 = float(al_M1[y == 0].mean())

    # ------------------------------------------------------------ E1
    if "E1" in only:
        log("E1: RQ2/RQ3 kiem tra lai")
        MD.append("## E1. Kiem tra lai RQ2 va RQ3 (model da luu, seed 42)\n")
        MD.append(f"FPR khung cua M1 tai nguong van hanh: {f3(fpr_M1)}\n")
        MD.append("### E1.1 Recall theo tung tan cong tai cung FPR khung (Bang 4.5 bo sung)\n")
        rows, e1 = [], {}
        for tg in (0.05, fpr_M1):
            for k in ("M1", "M2", "M3"):
                s = B42[k]
                t = thr_fpr(s[y == 0], tg)
                al = s > t
                rb = recall_block(al, att)
                fp = float(al[y == 0].mean())
                allk = (int(al[y == 1].sum()), int(y.sum()))
                rows.append([f3(tg), k, f3(fp), rec_str(allk)] + [rec_str(rb[x]) for x in ATTACKS])
                e1[f"{k}@{tg:.3f}"] = {"fpr": fp, "all": allk, **rb}
        md_table(["FPR muc tieu", "Cau hinh", "FPR dat", "Recall tong"] + ATTACKS, rows)
        OUT["E1.recall_matched"] = e1

        MD.append("### E1.2 H3: FPR khung du doan tu phan phoi N\n")
        yflow = att_flow != "benign"
        a_flow = float((s42["M1"][~yflow] >= 1).mean())
        a_net = float((s42["net"][~yflow & ~np.isnan(s42["net"])] >= 1).mean())
        a_proc = float((s42["proc"][~yflow & ~np.isnan(s42["proc"])] >= 1).mean())
        Nb = pd.Series(1, index=ws).groupby(level=0).size().reindex(bins_ws).fillna(0).to_numpy()
        Nben = Nb[y == 0]
        a_tgt = 1 - (1 - (100 - a.net_pct) / 100) * (1 - (100 - a.proc_pct) / 100)
        pred = lambda al_: float(np.mean(1 - (1 - al_) ** Nben))
        rows = [["alpha do duoc (cua so-luong lanh tinh, M1)", f3(a_flow)],
                ["alpha do duoc goc nhin mang / tien trinh", f"{f3(a_net)} / {f3(a_proc)}"],
                ["alpha muc tieu 1-(1-a_net)(1-a_proc)", f3(a_tgt)],
                ["N trung vi / trung binh / p90 (khung lanh tinh)",
                 f"{np.median(Nben):.0f} / {f2(np.mean(Nben))} / {np.percentile(Nben, 90):.0f}"],
                ["FPR khung thuc do (M1)", f3(fpr_M1)],
                ["Du doan 1-(1-a)^N voi N trung vi, a do duoc", f3(1 - (1 - a_flow) ** np.median(Nben))],
                ["Du doan 1-(1-a)^N voi N trung binh, a do duoc", f3(1 - (1 - a_flow) ** np.mean(Nben))],
                ["Du doan E[1-(1-a)^N] tren phan phoi N, a do duoc", f3(pred(a_flow))],
                ["Du doan E[1-(1-a)^N] tren phan phoi N, a muc tieu", f3(pred(a_tgt))]]
        md_table(["Dai luong", "Gia tri"], rows)
        OUT["E1.H3"] = {"alpha_flow": a_flow, "alpha_net": a_net, "alpha_proc": a_proc, "alpha_target": a_tgt,
                        "N_median": float(np.median(Nben)), "N_mean": float(np.mean(Nben)),
                        "fpr_bin": fpr_M1, "pred_E_measured": pred(a_flow), "pred_E_target": pred(a_tgt)}

        MD.append("### E1.3 Cung phan vi nguong cho hai goc nhin (go nhieu A5)\n")
        rows = []
        for pn, pp in ((a.net_pct, a.proc_pct), (99.5, 99.5), (99.9, 99.9)):
            tn, tp_ = float(np.percentile(val_err["net"], pn)), float(np.percentile(val_err["proc"], pp))
            zn, zp = s42["net"] * vm["net"].thr / tn, s42["proc"] * vm["proc"].thr / tp_
            m1 = np.nanmax(np.where(np.isnan(np.stack([zn, zp], 1)), -np.inf, np.stack([zn, zp], 1)), 1)
            b = to_bin(ws, np.where(np.isfinite(m1), m1, np.nan), bins_ws)
            al = to_bin(ws, (m1 >= 1).astype(float), bins_ws) > 0
            aucv = roc_auc_score(y, b)
            fpn = float((to_bin(ws, (zn >= 1).astype(float), bins_ws) > 0)[y == 0].mean())
            fpp = float((to_bin(ws, (zp >= 1).astype(float), bins_ws) > 0)[y == 0].mean())
            rb = recall_block(al, att)
            rows.append([f"{pn}/{pp}", f3(aucv), f3(al[y == 1].mean()), f3(al[y == 0].mean()),
                         f"{f3(fpn)} / {f3(fpp)}"] + [f3(rb[x][0] / max(1, rb[x][1])) for x in ATTACKS])
            OUT[f"E1.pct_{pn}_{pp}"] = {"auc": aucv, "recall": float(al[y == 1].mean()),
                                        "fpr": float(al[y == 0].mean()), "fpr_net": fpn, "fpr_proc": fpp}
        md_table(["Phan vi mang/tien trinh", "AUC M1", "Recall", "FPR khung", "FPR khung mang / tien trinh"]
                 + ATTACKS, rows)

    # ------------------------------------------------------------ E2
    m4_42 = None
    if "E2" in only or "E4" in only:
        per_seed = {}
        run_seeds = seeds if "E2" in only else [42]
        for sd in run_seeds:
            log(f"E2: seed {sd}")
            vv = {}
            for v, df in (("net", bn), ("proc", bp)):
                vv[v], _ = train_view_cols(df, Q.COLS[v], v, pct[v], sd, a)
            m4 = train_joint(Jb, {"net": NET, "proc": PROC}, sd, a.joint_pct, a)
            if sd == 42:
                m4_42 = m4
            s = score(J, vv, m4)
            Bs = {k: to_bin(ws, s[k], bins_ws) for k in ("M1", "M2", "M3", "M4")}
            alm1 = to_bin(ws, (s["M1"] >= 1).astype(float), bins_ws) > 0
            alm4 = to_bin(ws, (s["M4"] >= m4["tau"]).astype(float), bins_ws) > 0
            fp1 = float(alm1[y == 0].mean())
            r = {"thr_net": vv["net"].thr, "thr_proc": vv["proc"].thr, "fpr_M1": fp1,
                 "recall_M1": float(alm1[y == 1].mean()), "fpr_M4_own": float(alm4[y == 0].mean()),
                 "recall_M4_own": float(alm4[y == 1].mean())}
            for k, b in Bs.items():
                r[f"auc_{k}"] = float(roc_auc_score(y, b))
                for tg, nm in ((0.05, "r05"), (0.089, "r089")):
                    t = thr_fpr(b[y == 0], tg)
                    al = b > t
                    r[f"{nm}_{k}"] = float(al[y == 1].mean())
                    for x in ATTACKS:
                        m = att == x
                        r[f"{nm}_{k}_{x}"] = float(al[m].mean()) if m.any() else float("nan")
            per_seed[sd] = r
            if sd == 42:
                OUT["E2.seed42_bins"] = {k: v.tolist() for k, v in Bs.items()}
                B42.update({"M4": Bs["M4"]})
                B42_rt = Bs
        OUT["E2.per_seed"] = per_seed
        if "E2" in only:
            MD.append("## E2. Nhieu seed va bootstrap khoi co ghep cap\n")
            MD.append(f"Seed 42 tai hien model da luu: thr_net {f3(per_seed[42]['thr_net'])} so voi "
                      f"{f3(vm['net'].thr)}, thr_proc {f3(per_seed[42]['thr_proc'])} so voi {f3(vm['proc'].thr)}.\n")
            keys = ["auc_M1", "auc_M2", "auc_M3", "auc_M4", "r05_M1", "r05_M2", "r05_M3", "r05_M4",
                    "r089_M1", "r089_M4", "fpr_M1", "recall_M1", "fpr_M4_own", "recall_M4_own"] + \
                   [f"r089_{m}_{x}" for m in ("M1", "M3", "M4") for x in ATTACKS]
            rows = []
            for k in keys:
                vals = np.array([per_seed[s_][k] for s_ in per_seed])
                rows.append([k, f"{f3(np.nanmean(vals))} ± {f3(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0)}",
                             f"{f3(np.nanmin(vals))} – {f3(np.nanmax(vals))}",
                             " ".join(f3(v) for v in vals)])
            md_table(["Chi so", "Trung binh ± DLC", "Khoang", "Tung seed"], rows)

            MD.append("### Bootstrap khoi co ghep cap (seed 42, cung tap khung)\n")
            rows = []
            for A_, B_ in (("M1", "M4"), ("M1", "M3"), ("M1", "M2")):
                sa, sb = B42_rt[A_], B42_rt[B_]
                d0 = roc_auc_score(y, sa) - roc_auc_score(y, sb)
                fn = lambda i, sa=sa, sb=sb: roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i])
                lo, hi, v1 = boot(y, fn, a.n_boot, 42, a.block)
                lo2, hi2, _ = boot(y, fn, a.n_boot, 42, None)
                p = float(min(1.0, 2 * min((v1 <= 0).mean(), (v1 >= 0).mean())))
                rows.append([f"AUC {A_} − AUC {B_}", f3(d0), ci(lo, hi, False), ci(lo2, hi2, False), f3(p)])
                OUT[f"E2.dAUC_{A_}_{B_}"] = {"delta": d0, "block_ci": [lo, hi], "iid_ci": [lo2, hi2], "p_block": p}
            md_table(["So sanh", "Chenh lech", "KTC 95% bootstrap khoi", "KTC 95% bootstrap i.i.d.",
                      "p (hai phia, khoi)"], rows)
            MD.append("### KTC cua M1 (model da luu): bootstrap khoi so voi Wilson / i.i.d.\n")
            rows = []
            s1 = B42["M1"]
            lo, hi, _ = boot(y, lambda i: roc_auc_score(y[i], s1[i]), a.n_boot, 7, a.block)
            lo2, hi2, _ = boot(y, lambda i: roc_auc_score(y[i], s1[i]), a.n_boot, 7, None)
            rows.append(["ROC-AUC", f3(roc_auc_score(y, s1)), ci(lo, hi), ci(lo2, hi2)])
            for nm, msk in (("Recall", y == 1), ("FPR", y == 0)):
                fn = lambda i, msk=msk: float(al_M1[i][msk[i]].mean())
                lo, hi, _ = boot(y, fn, a.n_boot, 7, a.block)
                k_, n_ = int(al_M1[msk].sum()), int(msk.sum())
                rows.append([nm, f3(k_ / n_), ci(lo, hi), ci(*Q.wilson(k_, n_)) + " (Wilson)"])
            md_table(["Chi so M1", "Gia tri", "KTC bootstrap khoi", "KTC i.i.d."], rows)

    # ------------------------------------------------------------ E3
    if "E3" in only:
        log("E3: bo uid, distinct_binaries")
        MD.append("## E3. Bo dac trung danh tinh (uid, distinct_binaries)\n")
        pnoid = [c for c in PROC if c not in IDENTITY]
        vp, _ = train_view_cols(bp, pnoid, "proc", a.proc_pct, 42, a)
        vv = {"net": vm["net"], "proc": vp}
        m4n = train_joint(Jb, {"net": NET, "proc": pnoid}, 42, a.joint_pct, a)
        s = score(J, vv, m4n)
        rows = []
        base = {"M1 (co uid, da luu)": (B42["M1"], al_M1)}
        base["Chi tien trinh (co uid)"] = (B42["proc"], to_bin(ws, (s42["proc"] >= 1).astype(float), bins_ws) > 0)
        nb_ = {"M1 khong uid": s["M1"], "Chi tien trinh khong uid": s["proc"]}
        for nm, sc_ in nb_.items():
            b = to_bin(ws, sc_, bins_ws)
            base[nm] = (b, to_bin(ws, (sc_ >= 1).astype(float), bins_ws) > 0)
        b4 = to_bin(ws, s["M4"], bins_ws)
        base["M4 khong uid (nguong rieng)"] = (b4, to_bin(ws, (s["M4"] >= m4n["tau"]).astype(float), bins_ws) > 0)
        for nm, (b, al) in base.items():
            rb = recall_block(al, att)
            rows.append([nm, f3(roc_auc_score(y, b)), f3(al[y == 1].mean()), f3(al[y == 0].mean())]
                        + [f3(rb[x][0] / max(1, rb[x][1])) for x in ATTACKS])
        md_table(["Cau hinh", "AUC", "Recall", "FPR"] + ATTACKS, rows)
        MD.append("Tai cung FPR khung voi M1 da luu (" + f3(fpr_M1) + "):\n")
        rows = []
        for nm, b in (("M1 khong uid", to_bin(ws, s["M1"], bins_ws)), ("M4 khong uid", b4),
                      ("M3 khong uid", to_bin(ws, s["M3"], bins_ws))):
            t = thr_fpr(b[y == 0], fpr_M1)
            al = b > t
            rb = recall_block(al, att)
            rows.append([nm, f3(al[y == 1].mean())] + [f3(rb[x][0] / max(1, rb[x][1])) for x in ATTACKS])
        md_table(["Cau hinh", "Recall"] + ATTACKS, rows)
        # dac trung dong gop lon nhat sau khi bo uid
        Jr = J.reset_index(drop=True)
        rows = []
        for atk in ("slowloris", "brute"):
            sel = Jr[(att_flow == atk) & Jr.has_proc.to_numpy()]
            if len(sel) == 0:
                continue
            zp = s["proc"][sel.index.to_numpy()]
            top = pd.Series(zp, index=sel.index).groupby(sel["window_start"].to_numpy()).idxmax().to_numpy()
            X = Jr.loc[top, [f"proc__{c}" for c in pnoid]].to_numpy(np.float64)
            sq = vp.sq_errors(X)
            order = np.argsort(-sq, 1)[:, :3]
            names, cnt = np.unique(np.array(pnoid)[order].ravel(), return_counts=True)
            rows.append([atk, ", ".join(f"{n} ({c})" for n, c in sorted(zip(names, cnt), key=lambda x: -x[1])[:5])])
        md_table(["Tan cong", "Dac trung trong top-3 dong gop (so khung)"], rows)
        OUT["E3"] = {"rows": [r for r in rows]}

    # ------------------------------------------------------------ E3S: bo uid, nhieu seed
    if "E3S" in only:
        e3s = [int(x) for x in a.e3_seeds.split(",")]
        pnoid = [c for c in PROC if c not in IDENTITY]
        CFG = ["M1 co uid", "M1 khong uid", "Chi tien trinh khong uid", "M4 co uid", "M4 khong uid", "M3 khong uid"]
        res = {c: [] for c in CFG}
        for sd in e3s:
            log(f"E3S: seed {sd}")
            vn, _ = train_view_cols(bn, NET, "net", a.net_pct, sd, a)
            vp, _ = train_view_cols(bp, PROC, "proc", a.proc_pct, sd, a)
            vq, _ = train_view_cols(bp, pnoid, "proc", a.proc_pct, sd, a)
            m4u = train_joint(Jb, {"net": NET, "proc": PROC}, sd, a.joint_pct, a)
            m4n = train_joint(Jb, {"net": NET, "proc": pnoid}, sd, a.joint_pct, a)
            su = score(J, {"net": vn, "proc": vp}, m4u)
            sn = score(J, {"net": vn, "proc": vq}, m4n)
            own = {"M1 co uid": (su["M1"], su["M1"] >= 1), "M1 khong uid": (sn["M1"], sn["M1"] >= 1),
                   "Chi tien trinh khong uid": (sn["proc"], sn["proc"] >= 1),
                   "M4 co uid": (su["M4"], su["M4"] >= m4u["tau"]), "M4 khong uid": (sn["M4"], sn["M4"] >= m4n["tau"]),
                   "M3 khong uid": (sn["M3"], None)}
            fpr_ref = None
            for c in CFG:
                sc_, al_ = own[c]
                b = to_bin(ws, sc_, bins_ws)
                r = {"seed": sd, "auc": float(roc_auc_score(y, b))}
                if al_ is not None:
                    al = to_bin(ws, al_.astype(float), bins_ws) > 0
                    r["fpr"] = float(al[y == 0].mean()); r["recall"] = float(al[y == 1].mean())
                    for x in ATTACKS:
                        r[x] = float(al[att == x].mean())
                    if c == "M1 co uid":
                        fpr_ref = r["fpr"]
                # cung FPR voi M1 co uid cua chinh seed nay
                t = thr_fpr(b[y == 0], fpr_ref)
                am = b > t
                r["fpr_m"] = float(am[y == 0].mean()); r["recall_m"] = float(am[y == 1].mean())
                for x in ATTACKS:
                    r[f"{x}_m"] = float(am[att == x].mean())
                res[c].append(r)
        OUT["E3S"] = res
        ms = lambda v: f"{f3(np.mean(v))} ± {f3(np.std(v, ddof=1) if len(v) > 1 else 0)}"
        MD.append(f"## E3S. Bo uid, distinct_binaries — {len(e3s)} seed {e3s}\n")
        MD.append("### Tai nguong van hanh rieng (trung binh ± DLC qua cac seed)\n")
        rows = []
        for c in CFG:
            R_ = res[c]
            if "recall" not in R_[0]:
                continue
            rows.append([c, ms([r["auc"] for r in R_]), ms([r["recall"] for r in R_]), ms([r["fpr"] for r in R_])]
                        + [ms([r[x] for r in R_]) for x in ATTACKS])
        md_table(["Cau hinh", "AUC", "Recall", "FPR"] + ATTACKS, rows)
        MD.append("### Tai cung FPR khung voi M1 co uid cua tung seed\n")
        rows = []
        for c in CFG:
            R_ = res[c]
            rows.append([c, ms([r["auc"] for r in R_]), ms([r["recall_m"] for r in R_])]
                        + [ms([r[f"{x}_m"] for r in R_]) for x in ATTACKS])
        md_table(["Cau hinh", "AUC", "Recall", ] + ATTACKS, rows)
        MD.append("### Slowloris theo tung seed (nguong rieng / cung FPR)\n")
        rows = []
        for c in CFG:
            rows.append([c] + [f"{f3(r.get('slowloris', float('nan')))} / {f3(r['slowloris_m'])}" for r in res[c]])
        md_table(["Cau hinh"] + [f"seed {sd}" for sd in e3s], rows)

    # ------------------------------------------------------------ E4
    if "E4" in only:
        log("E4: tiem bat thuong co kiem soat")
        if m4_42 is None:
            m4_42 = train_joint(Jb, {"net": NET, "proc": PROC}, 42, a.joint_pct, a)
        rng = np.random.default_rng(2024)
        both = Jb[Jb.has_net & Jb.has_proc].reset_index(drop=True)
        if len(both) > a.inj_base:
            both = both.iloc[rng.choice(len(both), a.inj_base, replace=False)].reset_index(drop=True)
        Xn = both[[f"net__{c}" for c in NET]].to_numpy(np.float64)
        Xp = both[[f"proc__{c}" for c in PROC]].to_numpy(np.float64)
        Zn = vm["net"].scaler.transform(np.minimum(Xn, vm["net"].caps)).astype(np.float32)
        Zp = vm["proc"].scaler.transform(np.minimum(Xp, vm["proc"].caps)).astype(np.float32)
        Z4 = m4_42["prep"].transform(both).astype(np.float32)
        dn, dp = len(NET), len(PROC)
        elig = {}
        for v, Zv, cols in (("net", Zn, NET), ("proc", Zp, PROC)):
            elig[v] = [j for j, c in enumerate(cols) if c not in IDENTITY and Zv[:, j].std() > 1e-3]
        MD.append("## E4. Tiem bat thuong co kiem soat\n")
        MD.append(f"{len(both)} cua so-luong lanh tinh co du hai goc nhin. Tiem +delta (don vi do lech chuan, "
                  f"trong khong gian da chuan hoa cua tung mo hinh) vao k dac trung ngau nhien cua MOT goc nhin "
                  f"(bo uid, distinct_binaries va dac trung hang so). Dac trung du dieu kien: "
                  f"mang {len(elig['net'])}, tien trinh {len(elig['proc'])}.\n")

        def run_models(zn, zp, z4):
            en = Q.sq_err_model(vm["net"].model, zn).mean(1)
            ep = Q.sq_err_model(vm["proc"].model, zp).mean(1)
            e4 = Q.sq_err_model(m4_42["model"], z4).mean(1)
            return {"M1": np.maximum(en / vm["net"].thr, ep / vm["proc"].thr),
                    "M3": (dn * en + dp * ep) / (dn + dp), "M4": e4}

        base = run_models(Zn, Zp, Z4)
        deltas = [float(x) for x in a.inj_delta.split(",")]
        ks = [int(x) for x in a.inj_k.split(",")]
        fprs = [float(x) for x in a.inj_fpr.split(",")]
        thr = {(m, f): np.quantile(base[m], 1 - f) for m in base for f in fprs}
        own = {"M1": 1.0, "M4": m4_42["tau"]}
        curves = {}
        for v in ("net", "proc"):
            for k in ks:
                if k > len(elig[v]):
                    continue
                R_ = rng.random((len(both), len(elig[v])))
                pick = np.array(elig[v])[np.argsort(R_, 1)[:, :k]]
                for d in deltas:
                    zn, zp, z4 = Zn.copy(), Zp.copy(), Z4.copy()
                    rows_i = np.repeat(np.arange(len(both)), k)
                    cols_i = pick.ravel()
                    if v == "net":
                        zn[rows_i, cols_i] += d
                        z4[rows_i, cols_i] += d
                    else:
                        zp[rows_i, cols_i] += d
                        z4[rows_i, dn + cols_i] += d
                    sc_ = run_models(zn, zp, z4)
                    for m in sc_:
                        for f in fprs:
                            curves.setdefault((v, k, m, f), []).append(float((sc_[m] > thr[(m, f)]).mean()))
                        if m in own:
                            curves.setdefault((v, k, m, "own"), []).append(float((sc_[m] >= own[m]).mean()))

        def d50(c):
            c = np.asarray(c)
            for i in range(1, len(c)):
                if c[i] >= 0.5 > c[i - 1]:
                    return deltas[i - 1] + (0.5 - c[i - 1]) * (deltas[i] - deltas[i - 1]) / (c[i] - c[i - 1])
            return float("nan") if c[-1] < 0.5 else deltas[0]

        OUT["E4.curves"] = {"|".join(map(str, k)): v for k, v in curves.items()}
        OUT["E4.deltas"] = deltas
        for f in fprs:
            MD.append(f"### Ty le phat hien tai FPR cua so-luong {f3(f)} (nguong dat tren tap goc chua tiem)\n")
            rows = []
            for v in ("net", "proc"):
                for k in ks:
                    if (v, k, "M1", f) not in curves:
                        continue
                    for m in ("M1", "M3", "M4"):
                        rows.append([v, k, m] + [f3(x) for x in curves[(v, k, m, f)]])
            md_table(["Goc nhin", "k", "Cau hinh"] + [f"δ={f2(d)}" for d in deltas], rows)
            MD.append(f"delta50 (do lech chuan) va kappa thuc nghiem tai FPR {f3(f)}; "
                      f"kappa_e = (delta50_X / delta50_M1)^2 quy ve don vi sai so binh phuong:\n")
            rows = []
            for v in ("net", "proc"):
                for k in ks:
                    if (v, k, "M1", f) not in curves:
                        continue
                    d1 = d50(curves[(v, k, "M1", f)])
                    r = [v, k, f2(d1)]
                    for m in ("M3", "M4"):
                        dm = d50(curves[(v, k, m, f)])
                        r += [f2(dm), f2(dm / d1 if d1 else float("nan")), f2((dm / d1) ** 2 if d1 else float("nan"))]
                    rows.append(r)
                    OUT[f"E4.d50|{v}|{k}|{f}"] = r
            md_table(["Goc nhin", "k", "δ50 M1", "δ50 M3", "M3/M1", "κ_e M3", "δ50 M4", "M4/M1", "κ_e M4"], rows)
        MD.append("### Tai nguong van hanh rieng (M1: z>=1; M4: tau p" + str(a.joint_pct) + ")\n")
        rows = []
        for v in ("net", "proc"):
            for k in ks:
                for m in ("M1", "M4"):
                    if (v, k, m, "own") in curves:
                        rows.append([v, k, m] + [f3(x) for x in curves[(v, k, m, "own")]])
        md_table(["Goc nhin", "k", "Cau hinh"] + [f"δ={f2(d)}" for d in deltas], rows)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            f = fprs[0]
            kk = [k for k in ks if (("net", k, "M1", f) in curves)][:2]
            fig, axs = plt.subplots(len(kk), 2, figsize=(10, 3.4 * len(kk)), squeeze=False)
            sty = {"M1": ("#2e7d62", "-", "M1 — hợp nhất muộn (max)"), "M3": ("#a8502f", "--", "M3 — trung bình theo số chiều"),
                   "M4": ("#4b3f9e", ":", "M4 — kết hợp sớm")}
            for r_, k in enumerate(kk):
                for c_, (v, title) in enumerate((("net", "Tiêm vào góc nhìn mạng"), ("proc", "Tiêm vào góc nhìn tiến trình"))):
                    ax = axs[r_][c_]
                    for m, (col, ls, lb) in sty.items():
                        if (v, k, m, f) in curves:
                            ax.plot(deltas, curves[(v, k, m, f)], ls, color=col, marker="o", ms=3, label=lb)
                    ax.set_title(f"{title}, k = {k}")
                    ax.set_xlabel("δ (độ lệch chuẩn)")
                    ax.set_ylabel("tỷ lệ phát hiện")
                    ax.set_ylim(-0.02, 1.02)
                    ax.grid(alpha=0.3)
            axs[0][0].legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(os.path.join(a.outdir, "fig_injection.png"), dpi=300)
            plt.close(fig)
        except Exception as ex:  # pragma: no cover
            MD.append(f"(khong ve duoc hinh: {ex})")

    # ------------------------------------------------------------ xuat
    open(os.path.join(a.outdir, "extra_results.md"), "w", encoding="utf-8").write("\n".join(MD) + "\n")

    def to_py(o):
        return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(a.outdir, "extra_results.json"), "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, ensure_ascii=False, indent=1, default=to_py)
    log(f"xong -> {a.outdir}/extra_results.md")


if __name__ == "__main__":
    main()
