# ebpf-autoencoder

Unsupervised network- and host-level intrusion detection for Kubernetes, built on
**eBPF telemetry** (Cilium/Hubble + Tetragon) and a **dual autoencoder** with late
fusion. The system learns what *benign* traffic and process behaviour look like, then
flags 5-second time windows whose reconstruction error is anomalous — no attack labels
are needed for training.

This repository accompanies a thesis/研究 project (the source comments and evaluation
scripts are in Vietnamese) and contains the full pipeline: data collection, feature
engineering, model training, a live detector with a dashboard, and a battery of
reproducibility/verification scripts.

## How it works

```
        eBPF sensors                     feature engineering              dual autoencoder
  ┌────────────────────┐        ┌────────────────────────────┐      ┌──────────────────┐
  │ Hubble  (L3/L4      │        │ aggregate_features.py       │      │ net  AE  ──┐     │
  │  network flows)     │──JSON─▶│  5s windows, 5-tuple flows, │─CSV─▶│            ├─max─▶ score ≥ thr?
  │ Tetragon (tcp_*     │        │  rolling-60s scan features  │      │ proc AE  ──┘     │
  │  kprobe events)     │        └────────────────────────────┘      └──────────────────┘
  └────────────────────┘                                              trained ONLY on benign
```

- **Network branch** (15 features from Hubble): *relational* signals — flow counts,
  protocol ratios, cross-namespace ratio, distinct destination ports/IPs, and two
  rolling-60s per-source counters that expose port/host scanning.
- **Process branch** (18–19 features from Tetragon): *volume* signals — byte/packet
  counts and rates, `sendmsg`/`connect`/`close` counts, distinct binaries, uid, etc.
- Each branch is a small symmetric MLP autoencoder trained **only on benign** data.
  Reconstruction error above a per-branch threshold (99.5th percentile of held-out
  benign error) means "anomalous".
- **Late fusion:** the two branch scores are combined (default `max`). A 5-second bin is
  alerted if *any* flow in the bin exceeds threshold.

A key design property is **feature parity between batch and streaming**: the live
detector reuses the exact batch feature code, so there is no train/serve skew.
`application/parity_check.py` proves this cell-by-cell.

## Repository layout

| Folder | What's inside |
|--------|---------------|
| [data-collect/](data-collect/) | Dataset link and the attack-generation script used to build the labelled evaluation capture (authorized lab / Kubernetes Goat environment). |
| [training/](training/) | `train.py` — trains one autoencoder per branch on benign features, emits model/scaler/threshold/winsor artifacts. |
| [models/](models/) | Pre-trained artifacts for both branches, ready to score with. |
| [application/](application/) | The runtime: feature aggregation (batch + streaming), the live FastAPI detector, the dashboard, and offline replay/parity tools. |
| [evaluate-and-verification/](evaluate-and-verification/) | Evaluation, ablation, feature-quality plots, and scripts that reproduce/verify every table and figure in the report. |

## Quick start

```bash
# 0. install deps
pip install torch numpy pandas scikit-learn matplotlib fastapi uvicorn

# 1. aggregate benign eBPF logs into per-branch feature CSVs
python3 application/aggregate_features.py \
    --hubble   /tmp/ebpf-logs/cilium-network.json \
    --tetragon /tmp/ebpf-logs/tetragon-events.json \
    --outdir   /tmp/ebpf-logs/features --window 5 --rolling 60

# 2. train one autoencoder per branch (benign only)
python3 training/train.py --branch net  --csv /tmp/ebpf-logs/features/features_network.csv --outdir models
python3 training/train.py --branch proc --csv /tmp/ebpf-logs/features/features_process.csv --outdir models

# 3. evaluate on a labelled attack capture
python3 evaluate-and-verification/evaluate.py \
    --net-csv  attack/features/features_network.csv \
    --proc-csv attack/features/features_process.csv \
    --net-model models/net_model.pt --net-scaler models/net_scaler.pkl --net-thr models/net_thr.txt \
    --proc-model models/proc_model.pt --proc-scaler models/proc_scaler.pkl --proc-thr models/proc_thr.txt \
    --labels labels.csv --outdir eval_out

# 4. run the live detector + dashboard
uvicorn detector_service:app --host 0.0.0.0 --port 8000   # from inside application/
```

## Data

The eBPF log captures and feature CSVs are hosted externally — see
[data-collect/README.MD](data-collect/README.MD) for the download link.

## Note on a companion module

`application/detector_service.py` and `application/replay_detect.py` import a
`detector_core` module (branch scoring + `fuse_rows`) that is **not committed** in this
repository. Restore it (or the equivalent `fusion.py` referenced throughout the
comments) alongside `application/` to run the live detector and replay tools. The batch
pipeline (`aggregate_features.py`, `train.py`, `evaluate.py`) is self-contained and runs
without it.

## Scope / ethics

The attack scripts under `data-collect/` are for generating a labelled dataset in an
**isolated, authorized lab** (a Kubernetes Goat cluster). Only run them against
infrastructure you own or are explicitly permitted to test.
