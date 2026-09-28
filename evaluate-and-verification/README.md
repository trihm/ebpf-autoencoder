# evaluate-and-verification

Everything used to measure the detector and to reproduce/verify the tables and figures in
the written report. All scripts share the same window-labelling rule: a `window_start`
that falls inside an attack's `[start, end]` interval (from `labels.csv`) is that attack;
everything else is benign. A 5-second bin is alerted if **any** flow in it exceeds
threshold.

Inputs are generally the same set: the attack feature CSVs (network + process), the six
model artifacts from [../models/](../models/), and `labels.csv`.

## Scripts

| Script | What it produces / verifies |
|--------|------------------------------|
| `evaluate.py` | Main evaluation. Scores each branch, fuses (`max`), labels windows, and reports global ROC-AUC, precision/recall/F1 at threshold, **per-attack recall** (nmap / hping3 / slowloris / brute), and an Isolation Forest baseline. Exports 300-DPI ROC curve and error-timeline figures. |
| `extra_experiments_3s.py` | Supplementary experiments (imports `train.py`): recheck per-attack recall at fixed FPRs with Wilson intervals (E1); multi-seed + paired moving-block bootstrap on M1–M4 AUC/recall (E2); ablation dropping identity features `uid`/`distinct_binaries` (E3); controlled anomaly injection to measure detection vs. injected delta (E4). |
| `plot_feature_quality.py` | Four 300-DPI feature-quality figures: per-feature benign-vs-attack histograms, per-attack branch "signature" (which features blow up), Spearman correlation heatmap (justifies bottleneck size), and single-feature AUC separability bars. |
| `validate_ablation.py` | Re-runs the bottleneck-size ablation and reports at the 5-second bin unit (reproduces the report's network/process ablation tables). Trains per (bottleneck, seed), averages over seeds, warns on high seed variance. |
| `validate_separation.py` | Pins down the exact "separation" formula used in the ablation table by retraining per bottleneck size and testing several candidate formulas against the printed values. |
| `validate_confusion_matrix.py` | Verifies the confusion matrix (report Table 4.2). Either recomputes from `scored_windows.csv`, or reconstructs it from reported `(n_attack, n_benign, recall, fpr)` and checks precision/F1 for consistency. Optionally renders the figure; exits non-zero on mismatch. |
| `validate_checklist.py` | Answers the pre-defense review checklist from data: actual `dropped_ratio` max/p99, final nmap recall, and the 10,000-port scan duration from `labels.csv`. Every input is optional; missing files are skipped. |

## Typical run

```bash
python3 evaluate.py \
    --net-csv  attack/features/features_network.csv \
    --proc-csv attack/features/features_process.csv \
    --net-model ../models/net_model.pt --net-scaler ../models/net_scaler.pkl --net-thr ../models/net_thr.txt \
    --proc-model ../models/proc_model.pt --proc-scaler ../models/proc_scaler.pkl --proc-thr ../models/proc_thr.txt \
    --labels labels.csv \
    --benign-net benign/features/features_network.csv \
    --benign-proc benign/features/features_process.csv \
    --outdir eval_out

# then verify the reported confusion matrix against the scored output
python3 validate_confusion_matrix.py --scored eval_out/scored_windows.csv \
    --expect-recall 0.829 --expect-fpr 0.089 --expect-precision 0.870 --expect-f1 0.849 \
    --fig confusion_matrix.png
```

## Notes

- Figures use ASCII titles (to avoid missing-glyph errors); the report's Vietnamese
  captions are added separately.
- Scripts that retrain (`extra_experiments_3s.py`, `validate_ablation.py`,
  `validate_separation.py`) mirror `train.py` exactly: symmetric MLP AE, `StandardScaler`
  fit on the train split, seed 42, 100 epochs, Adam lr 1e-3, batch 64.
- Some docstrings reference `evaluate_v3.py`/`fusion.py` (earlier names); the committed
  entry point is `evaluate.py`.
