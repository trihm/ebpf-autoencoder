# training

Trains the autoencoders. One branch at a time, **benign data only** — the model never
sees an attack during training. Run it twice to produce the dual-AE.

## `train.py`

```bash
# network branch (Hubble / relational features)
python3 train.py --branch net  --csv features_network.csv --outdir ../models

# process branch (Tetragon / volume features)
python3 train.py --branch proc --csv features_process.csv --outdir ../models
```

### What it does

1. **Load benign features** for the branch and fill NaN/inf with 0.
2. **Split** train / held-out (default 80/20).
3. **Winsorize** long-tailed rate/count columns at p99.9 (caps learned from the train
   split, saved to disk and re-applied identically at eval to avoid train/serve skew).
   The two rolling-60s scan features are deliberately **not** capped — their long tail
   *is* the scan signal.
4. **Standardize** (`StandardScaler` fit on the train split only).
5. **Train** a symmetric MLP autoencoder (bottleneck ≈ ⅓ of input width), 100 epochs,
   Adam, lr 1e-3, batch 64, seed 42.
6. **Threshold** = 99.5th percentile of reconstruction error on held-out benign.

### Model architecture (`Autoencoder`)

```
n_in ─▶ Linear(n_in→h1) ─ReLU─▶ Linear(h1→h2) ─ReLU─▶ Linear(h2→h1) ─ReLU─▶ Linear(h1→n_in)
        h1 = max(8, n_in·2/3)     h2 = max(4, n_in/3)  (bottleneck)
```

### Outputs (written to `--outdir`, one set per branch)

Filenames follow the contract the fusion/eval code expects, with `{prefix}` = `net` or `proc`:

| File | Contents |
|------|----------|
| `{prefix}_model.pt` | full model object (`torch.load(..., weights_only=False)`) |
| `{prefix}_scaler.pkl` | fitted `StandardScaler` (pickle) |
| `{prefix}_thr.txt` | single float: the 99.5th-percentile benign threshold |
| `{prefix}_winsor.npz` | winsor caps + feature column order, re-applied at eval |

### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--branch` | *(required)* | `net` or `proc` |
| `--csv` | *(required)* | benign feature CSV for this branch |
| `--epochs` | 100 | training epochs |
| `--val-frac` | 0.2 | held-out fraction for the threshold |
| `--percentile` | 99.5 | threshold percentile on held-out benign |
| `--winsor-pct` | 99.9 | cap percentile for rate/count columns |
| `--no-winsor` | off | disable winsorizing (for before/after comparison) |
| `--seed` | 42 | RNG seed (torch + numpy) |

The feature lists (`NETWORK_FEATURES`, `PROCESS_FEATURES`) are defined at the top of the
file and **must stay in sync** with `application/aggregate_features.py` and the fusion
code. If the input CSV is missing a column the run aborts; fewer than 50 benign rows
triggers an unstable-threshold warning.
