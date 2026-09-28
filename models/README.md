# models

Pre-trained dual-autoencoder artifacts, ready to score with. These were produced by
[training/train.py](../training/train.py) on the **benign** capture and are consumed by
the evaluation and live-detector code.

## Files

Two branches, `net` (network / Hubble) and `proc` (process / Tetragon), each with four
artifacts:

| File | Type | Purpose |
|------|------|---------|
| `net_model.pt` / `proc_model.pt` | PyTorch model | trained autoencoder (load with `torch.load(..., weights_only=False)`) |
| `net_scaler.pkl` / `proc_scaler.pkl` | pickled `StandardScaler` | feature standardization fit on benign train split |
| `net_thr.txt` / `proc_thr.txt` | text (one float) | reconstruction-error threshold (99.5th percentile of held-out benign) |
| `net_winsor.npz` / `proc_winsor.npz` | numpy archive | winsor caps + feature column order, re-applied at scoring time |

## Current thresholds

| Branch | Threshold |
|--------|-----------|
| net  | `0.5054380894` (`net_thr.txt`) |
| proc | `0.3294960558` (`proc_thr.txt`) |

## Usage contract

All four files of a branch must be used together and applied in this order when scoring:
**winsorize → scale → autoencoder → compare reconstruction MSE against the threshold.**
Mixing artifacts from different training runs, or skipping the winsor caps, produces
train/serve skew and invalid scores.

> These are binary/pickle artifacts. Loading a pickle executes code — only load model
> files you trust. Regenerate them from source with `training/train.py` if in doubt.
