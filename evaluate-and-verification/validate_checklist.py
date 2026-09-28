#!/usr/bin/env python3
"""
Validate checklist tham dinh truoc bao ve. Tra loi bang DU LIEU cho cac muc
co the tu dong hoa:

  Muc 1: dropped_ratio trong Bang 3.4 — Max & P99 thuc te la bao nhieu?
  Muc 2: Recall nmap cuoi cung — 0,143 hay 0,214? (chot theo scored moi nhat)
  Muc 3: Thoi luong quet 10.000 cong — moc bat dau/ket thuc trong labels.csv?

Cac muc con lai (4, 5, 6) can thao tac khac — xem phan HUONG DAN o cuoi khi chay.

Chay:
  python3 validate_checklist.py \
      --benign-net benign/features/features_network.csv \
      --labels labels.csv \
      --scored eval_out/scored_windows.csv \
      --window 5

Moi tham so deu tuy chon: muc nao thieu file thi script bao va bo qua.
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


def rfc3339_epoch(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def muc1_dropped_ratio(benign_net):
    print("=" * 70)
    print("MUC 1: dropped_ratio trong Bang 3.4 — Max & P99 thuc te")
    print("=" * 70)
    if benign_net is None:
        print("  [bo qua] thieu --benign-net\n"); return
    df = pd.read_csv(benign_net)
    if "dropped_ratio" not in df.columns:
        print("  [!] khong co cot dropped_ratio\n"); return
    v = pd.to_numeric(df["dropped_ratio"], errors="coerce").dropna()
    pct = 100.0 * (v != 0).mean()
    print(f"  n = {len(v)} cua so benign")
    print(f"  %!=0      = {pct:.3f}%   ({int((v!=0).sum())} cua so khac 0)")
    print(f"  trung vi  = {np.median(v):.4f}")
    print(f"  P99       = {np.percentile(v,99):.4f}")
    print(f"  P99.5     = {np.percentile(v,99.5):.4f}")
    print(f"  Max       = {v.max():.4f}")
    print()
    print("  KET LUAN: neu %!=0 > 0 thi Max PHAI > 0. Bang ghi 'Max=0' voi '0,1% khac 0'")
    print("  la MAU THUAN noi tai. Sua Bang 3.4: dien Max & P99 thuc te o tren.")
    print("  (Neu P99=0 nhung Max>0: dung, vi <1% cua so khac 0 nen P99 van roi vao 0.)\n")


def muc2_recall_nmap(scored, window):
    print("=" * 70)
    print("MUC 2: Recall nmap cuoi cung — 0,143 hay 0,214?")
    print("=" * 70)
    if scored is None:
        print("  [bo qua] thieu --scored\n"); return
    df = pd.read_csv(scored)
    sub = df[df["attack"] == "nmap_synscan"]
    if len(sub) == 0:
        print("  [!] khong co window nmap trong scored\n"); return
    # time-bin recall: gom theo window_start, max score
    tb = sub.groupby("window_start").agg(mx=("score", "max")).reset_index()
    nbin = len(tb)
    alert = int((tb["mx"] >= 1.0).sum())
    rec = alert / nbin if nbin else float("nan")
    # per-flow recall (de doi chieu, KHONG dung cho luan van)
    flow_rec = float((sub["alert"] == True).mean()) if "alert" in sub.columns else float("nan")
    print(f"  time-bin: {alert}/{nbin} = {rec:.3f}  <-- CON SO CHOT (dung cho luan van)")
    print(f"  per-flow: {flow_rec:.4f}  (KHONG dung — bi lech boi so luong flow)")
    print()
    print(f"  KET LUAN: recall nmap = {rec:.3f} theo time-bin (don vi IDS chuan).")
    print("  Thong nhat con so nay o Bang 4.1, muc 4.4 va 5.2. Con so 0,214 (neu co)")
    print("  la ban CU truoc khi sua bug ephemeral — thay bang con so tren.\n")


def muc3_scan_duration(labels, window):
    print("=" * 70)
    print("MUC 3: Thoi luong quet 10.000 cong — moc trong labels.csv")
    print("=" * 70)
    if labels is None:
        print("  [bo qua] thieu --labels\n"); return
    ldf = pd.read_csv(labels)
    nmap_rows = ldf[ldf["attack"].str.contains("nmap", case=False, na=False)]
    if len(nmap_rows) == 0:
        print("  [!] khong tim thay dong nmap trong labels.csv"); 
        print("  Cac attack co:", list(ldf["attack"].unique()), "\n"); return
    for _, r in nmap_rows.iterrows():
        s = rfc3339_epoch(r["start"]); e = rfc3339_epoch(r["end"])
        dur = e - s
        print(f"  {r['attack']}: start={r['start']}  end={r['end']}")
        print(f"    -> thoi luong = {dur:.1f} giay ({dur/60:.2f} phut)")
        n_bins = int(np.ceil(dur / window))
        print(f"    -> so time-bin ({window}s) phu = {n_bins}")
    print()
    print("  KET LUAN: thay '1-2 giay' o muc 4.4 va 5.2 bang thoi luong THUC o tren.")
    print("  Neu thoi luong ngan (vai giay) va window=5s -> chi vai bin, giai thich")
    print("  vi sao recall nmap thap (it bin de bat).\n")


def huong_dan_con_lai():
    print("=" * 70)
    print("CAC MUC CON LAI — huong dan (khong tu dong hoa duoc)")
    print("=" * 70)
    print("""
  MUC 4: AUC & recall khi CHI dung mot nhanh
    Can chay eval RIENG tung nhanh: dat score = norm_net (chi network) roi
    score = norm_proc (chi process), tinh lai ROC-AUC va recall time-bin.
    -> Neu ban co scored_windows.csv, co the tinh gan dung:
         chi_net:  alert khi norm_net >= 1  (bo qua norm_proc)
         chi_proc: alert khi norm_proc >= 1
       Roi so recall per-attack giua 3 cau hinh: net-only / proc-only / fusion.
    -> Muon chinh xac (ROC-AUC), can diem norm lien tuc tung nhanh, chay
       roc_auc_score(y_timebin, score_branch). Bao mình neu muon script rieng.

  MUC 5: Bang 3.7 — bottleneck 6 va 8 giong het nhau?
    Kiem tra config train.py / fusion.py: kich thuoc bottleneck thuc te la bao
    nhieu cho moi nhanh? Neu bang ghi 2 dong (bottleneck=6 va =8) co so LIET KE
    giong het -> hoac (a) nhap nham, hoac (b) hai cau hinh cho ket qua trung
    hop. Chay lai train voi --bottleneck 6 va --bottleneck 8, so val_loss /
    recall de xac nhan. 'Do tach biet' o Bang 3.6: hoi ro cong thuc — thuong la
    |AUC-0.5|+0.5 (nhu feat_univariate_auc) hoac khoang cach median chuan hoa.

  MUC 6: Hinh 3.2 ve lai theo so moi
    Da co plot_feature_quality.py. Chay:
      python3 plot_feature_quality.py \\
          --net-csv attack/.../features_network.csv \\
          --proc-csv attack/.../features_process.csv \\
          --labels labels.csv \\
          --benign-net benign/.../features_network.csv \\
          --benign-proc benign/.../features_process.csv \\
          --window 5 --outdir hinh32_moi
    Hinh feat_separation_grid.png la ban thay the Hinh 3.2 voi so moi.
""")


# ---------------------------------------------------------------------------
# MUC 4 (mo rong): recall per-attack theo 3 cau hinh net-only/proc-only/fusion
# Goi rieng: python3 validate_checklist.py --scored ... --branch-compare
# ---------------------------------------------------------------------------
def muc4_branch_compare(scored, window):
    print("=" * 70)
    print("MUC 4: Recall per-attack — net-only vs proc-only vs fusion")
    print("=" * 70)
    df = pd.read_csv(scored)
    attacks = [a for a in df["attack"].unique() if a != "benign"]
    print(f"  {'attack':16s} {'net-only':>9s} {'proc-only':>10s} {'fusion':>8s}")
    print("  " + "-" * 46)
    for atk in sorted(attacks):
        sub = df[df["attack"] == atk]
        tb = sub.groupby("window_start").agg(
            mn=("norm_net", "max"), mp=("norm_proc", "max"),
            mx=("score", "max")).reset_index()
        n = len(tb)
        r_net = (tb["mn"] >= 1.0).sum() / n
        r_proc = (tb["mp"] >= 1.0).sum() / n
        r_fus = (tb["mx"] >= 1.0).sum() / n
        print(f"  {atk:16s} {r_net:>9.3f} {r_proc:>10.3f} {r_fus:>8.3f}")
    print()
    print("  KET LUAN: cot fusion PHAI >= max(net-only, proc-only) neu late-fusion")
    print("  dung phep hop. Bang nay chinh la bang ablation cho Chuong 4 (muc 4).")
    print("  ROC-AUC tung nhanh can diem norm lien tuc — dung norm_net/norm_proc")
    print("  lam score lien tuc roi roc_auc_score(y_timebin, score).\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-net")
    ap.add_argument("--labels")
    ap.add_argument("--scored")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--branch-compare", action="store_true", help="chay bang so sanh nhanh (muc 4)")
    args = ap.parse_args()

    muc1_dropped_ratio(args.benign_net)
    muc2_recall_nmap(args.scored, args.window)
    muc3_scan_duration(args.labels, args.window)
    if args.branch_compare and args.scored:
        muc4_branch_compare(args.scored, args.window)
    huong_dan_con_lai()


if __name__ == "__main__":
    main()
