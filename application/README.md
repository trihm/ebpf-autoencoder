# application

The runtime side of the project: turn eBPF logs into features, score them with the
dual-AE, and serve a near-real-time detector with a dashboard. The same feature code
backs both the batch pipeline and the live stream, so there is **no train/serve skew**.

## Files

| File | Role |
|------|------|
| `aggregate_features.py` | **Batch** feature engineering. Reads Hubble + Tetragon JSON, emits `features_network.csv` (15 feats) and `features_process.csv` (18–19 feats) keyed by `(flow_key, window_start)`. |
| `stream_features.py` | **Streaming** wrapper around the *exact same* aggregation logic — an in-RAM rolling buffer that calls `aggregate_features` per window once it's mature. Also the NDJSON parsers (`parse_hubble_obj`, `parse_tetragon_obj`), `StreamAggregator`, and `LogTailer`. |
| `detector_service.py` | **Live detector** — FastAPI app: tail logs → `StreamAggregator` (5s windows) → dual-AE score + fusion → group into incidents → push to the browser over SSE. Persists alerts to SQLite. |
| `dashboard.html` | Browser dashboard (Chart.js): live stat cards, an error timeline, and an incident table fed by the SSE stream. |
| `replay_detect.py` | Run the **whole live path** (stream → score → fuse) offline on a saved, labelled attack log — to confirm the live path reproduces the offline evaluation numbers. |
| `parity_check.py` | Proof that **live features == batch features**, cell by cell. Run this before trusting any number the live detector reports. |

## Feature engineering details

- **Flow unit:** normalized bidirectional 5-tuple, bucketed into **5-second windows**
  (`--window 5`). IPv4-mapped IPv6 prefixes are stripped to avoid flow fragmentation.
- **Rolling scan features:** each window also carries two per-source **rolling-60s**
  counters (`distinct_dst_ports_60s`, `distinct_dst_ips_60s`) that expose port/host
  scanning. These are the features that catch `nmap`.
- **Two CSVs, not one join:** the network and process branches are scored independently
  and their scores are combined by late fusion (default `max`) — not joined at the
  feature level.

## Running the live detector

```bash
# from inside application/  (aggregate_features.py, stream_features.py, detector_core on PYTHONPATH)
uvicorn detector_service:app --host 0.0.0.0 --port 8000
# then open dashboard.html (served at the root route)
```

Configuration is via environment variables (with sensible defaults in the file):
`HUBBLE_LOG`, `TETRAGON_LOG`, `MODEL_DIR`, `WINDOW`, `ROLLING`, `GRACE`, `FUSION_MODE`,
`INCIDENT_GAP`, `DB_PATH`.

## Offline sanity checks

```bash
# 1. prove no train/serve skew (should report 0 differing cells)
python3 parity_check.py --hubble cilium-network.json --tetragon tetragon-events.json

# 2. replay the live path on a labelled attack capture
python3 replay_detect.py \
    --hubble attack-network.json --tetragon attack-tetragon.json \
    --labels labels.csv --model-dir ../models --out live_scored.csv
```

## Companion module (not committed)

`detector_service.py` and `replay_detect.py` import **`detector_core`**
(`NETWORK_FEATURES`, `PROCESS_FEATURES`, `Branch`, `fuse_rows`, `src_ip_of`) — the
branch scorer and fusion logic, referred to as `fusion.py` in the comments. This module
is **not present** in the repository and must be restored on the `PYTHONPATH` for the
live detector and replay tools to run. `aggregate_features.py`, `stream_features.py`, and
`parity_check.py` run without it.
