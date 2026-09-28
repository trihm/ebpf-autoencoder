#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_confusion_matrix.py — kiem chung ma tran nham lan (confusion matrix)
cho Bang 4.2 cua do an, va sinh hinh de chen vao bao cao.

HAI CHE DO:

1) --scored eval_out/scored_windows.csv   (KHUYEN DUNG - so lieu that)
   Doc lai file ma evaluate_v3.py da xuat, gom flow ve time-bin 5s dung y HET
   logic cua evaluate_v3.py:
       tb = m.groupby("window_start").agg(score=("score","max"), y=("y","max"))
       alert = score >= 1.0
   Roi in ra TP/FN/FP/TN + Precision/Recall/F1/FPR va doi chieu voi so trong
   bao cao (--expect-*). Thoat ma 1 neu lech.

2) --from-metrics  (khong can file, dung khi chi con so trong bao cao)
   Suy nguoc ma tran tu (n_attack, n_benign, recall, fpr) roi kiem tra lai
   precision / F1 co khop voi bao cao khong. Dung de phat hien so lieu
   khong nhat quan giua cac bang.

Sinh hinh:  them --fig confusion_matrix.png  (dung duoc o ca hai che do)

Vi du:
  python3 validate_confusion_matrix.py --scored eval_out/scored_windows.csv \
      --expect-recall 0.829 --expect-fpr 0.089 --expect-precision 0.870 \
      --expect-f1 0.849 --fig confusion_matrix.png

  python3 validate_confusion_matrix.py --from-metrics --n-attack 170 \
      --n-benign 236 --recall 0.829 --fpr 0.089 \
      --expect-precision 0.870 --expect-f1 0.849 --fig confusion_matrix.png
"""
import argparse
import sys


def metrics_from_cm(tp, fn, fp, tn):
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    acc = (tp + tn) / (tp + fn + fp + tn)
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "accuracy": acc}


def cm_from_scored(path, threshold=1.0):
    import pandas as pd
    m = pd.read_csv(path)
    for col in ("window_start", "score", "y"):
        if col not in m.columns:
            sys.exit(f"[loi] {path} thieu cot '{col}'. "
                     f"Cac cot dang co: {list(m.columns)}")
    tb = (m.groupby("window_start")
            .agg(score=("score", "max"), y=("y", "max"))
            .reset_index())
    tb["alert"] = tb["score"] >= threshold
    tp = int(((tb["y"] == 1) & tb["alert"]).sum())
    fn = int(((tb["y"] == 1) & ~tb["alert"]).sum())
    fp = int(((tb["y"] == 0) & tb["alert"]).sum())
    tn = int(((tb["y"] == 0) & ~tb["alert"]).sum())
    print(f"[nguon] {path}: {len(m)} dong luong-cua so -> {len(tb)} time-bin")
    return tp, fn, fp, tn


def cm_from_metrics(n_attack, n_benign, recall, fpr):
    tp = round(recall * n_attack)
    fn = n_attack - tp
    fp = round(fpr * n_benign)
    tn = n_benign - fp
    print(f"[nguon] suy nguoc tu chi so bao cao: "
          f"{n_attack} bin tan cong, {n_benign} bin lanh tinh")
    return tp, fn, fp, tn


def print_report(tp, fn, fp, tn, expects, tol):
    mt = metrics_from_cm(tp, fn, fp, tn)
    total = tp + fn + fp + tn
    print("\n=== MA TRAN NHAM LAN (don vi time-bin 5 giay) ===")
    print(f"{'':22s}{'Du doan: Tan cong':>20s}{'Du doan: Lanh tinh':>21s}")
    print(f"{'Thuc te: Tan cong':22s}{tp:>20d}{fn:>21d}")
    print(f"{'Thuc te: Lanh tinh':22s}{fp:>20d}{tn:>21d}")
    print(f"\nTong: {total} time-bin ({tp + fn} tan cong / {fp + tn} lanh tinh)")
    print(f"Precision {mt['precision']:.4f} | Recall {mt['recall']:.4f} | "
          f"F1 {mt['f1']:.4f} | FPR {mt['fpr']:.4f} | Accuracy {mt['accuracy']:.4f}")

    ok = True
    if expects:
        print("\n=== DOI CHIEU VOI SO TRONG BAO CAO ===")
        for name, want in expects.items():
            got = mt[name]
            good = abs(got - want) <= tol
            ok &= good
            print(f"  {name:10s} bao cao={want:.4f}  tinh lai={got:.4f}  "
                  f"lech={abs(got - want):.4f}  {'OK' if good else 'LECH'}")
    return mt, ok


def make_figure(tp, fn, fp, tn, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "DejaVu Sans"
    cm = np.array([[tp, fn], [fp, tn]], dtype=int)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = cm / row_sum * 100.0

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    # to mau theo ty le hang: o dung dam, o sai nhat
    shade = np.array([[pct[0, 0], pct[0, 1]], [pct[1, 0], pct[1, 1]]])
    mask_correct = np.array([[True, False], [False, True]])
    colors = np.zeros(cm.shape + (3,))
    for i in range(2):
        for j in range(2):
            t = shade[i, j] / 100.0
            if mask_correct[i, j]:          # o dung -> xanh duong
                colors[i, j] = (1 - 0.75 * t, 1 - 0.45 * t, 1 - 0.12 * t)
            else:                            # o sai -> do nhat
                colors[i, j] = (1 - 0.05 * t, 1 - 0.55 * t, 1 - 0.50 * t)
    ax.imshow(colors, aspect="auto")

    labels_pred = ["Dự đoán:\nTấn công", "Dự đoán:\nLành tính"]
    labels_true = ["Thực tế:\nTấn công", "Thực tế:\nLành tính"]
    names = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.16, f"{names[i][j]} = {cm[i, j]}",
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color="#10243e")
            ax.text(j, i + 0.16, f"{pct[i, j]:.1f}% của hàng".replace(".", ","),
                    ha="center", va="center", fontsize=10.5, color="#31425a")

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_pred, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_true, fontsize=11)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    mt = metrics_from_cm(tp, fn, fp, tn)
    vn = lambda v: f"{v:.3f}".replace(".", ",")
    ax.set_title(
        f"Precision {vn(mt['precision'])} · Recall {vn(mt['recall'])} · "
        f"F1 {vn(mt['f1'])} · FPR {vn(mt['fpr'])}",
        fontsize=11.5, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\n[hinh] da ghi {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", help="eval_out/scored_windows.csv tu evaluate_v3.py")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="nguong tren diem hop nhat da chuan hoa (mac dinh 1.0)")
    ap.add_argument("--from-metrics", action="store_true",
                    help="suy nguoc ma tran tu chi so thay vi doc CSV")
    ap.add_argument("--n-attack", type=int, default=170)
    ap.add_argument("--n-benign", type=int, default=236)
    ap.add_argument("--recall", type=float, default=0.829)
    ap.add_argument("--fpr", type=float, default=0.089)
    ap.add_argument("--expect-precision", type=float)
    ap.add_argument("--expect-recall", type=float)
    ap.add_argument("--expect-f1", type=float)
    ap.add_argument("--expect-fpr", type=float)
    ap.add_argument("--tol", type=float, default=0.001,
                    help="sai so cho phep khi doi chieu (mac dinh 0.001)")
    ap.add_argument("--fig", help="duong dan PNG de xuat hinh ma tran nham lan")
    args = ap.parse_args()

    if args.scored:
        tp, fn, fp, tn = cm_from_scored(args.scored, args.threshold)
    elif args.from_metrics:
        tp, fn, fp, tn = cm_from_metrics(args.n_attack, args.n_benign,
                                         args.recall, args.fpr)
    else:
        sys.exit("Can --scored <csv> hoac --from-metrics. Xem --help.")

    expects = {}
    for k in ("precision", "recall", "f1", "fpr"):
        v = getattr(args, f"expect_{k}")
        if v is not None:
            expects[k] = v
    if args.from_metrics:
        expects.setdefault("recall", args.recall)
        expects.setdefault("fpr", args.fpr)

    _, ok = print_report(tp, fn, fp, tn, expects, args.tol)

    if args.fig:
        make_figure(tp, fn, fp, tn, args.fig)

    if not ok:
        print("\n[!] Co chi so LECH qua nguong dung sai — kiem tra lai truoc khi nop.")
        return 1
    print("\n[OK] Ma tran nham lan nhat quan voi cac chi so da bao cao.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
