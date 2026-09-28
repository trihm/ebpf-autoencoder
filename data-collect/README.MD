# data-collect

Scripts and pointers for building the labelled dataset used to train and evaluate the
dual-autoencoder detector.

> ⚠️ The attack generator in this folder is for producing a labelled capture inside an
> **isolated, authorized lab** (a [Kubernetes Goat](https://madhuakula.com/kubernetes-goat/)
> cluster). Only run it against infrastructure you own or have explicit permission to test.

## Dataset

The raw eBPF log captures (Hubble network flows + Tetragon process events) and the
derived feature CSVs are hosted on Google Drive:

**https://drive.google.com/drive/folders/1YBcpmJuJxeoFgmcKlYoK64fjDgelis1Y?usp=sharing**

The dataset is split into a **benign** capture (used to train the models and set
thresholds) and a **attack** capture with a `labels.csv` timeline (used for evaluation).

## `run-attack.sh`

Drives a sequence of attacks from the host against services in the Kubernetes Goat
cluster, wrapping each one with host-clock start/end timestamps and auto-generating
`labels.csv` (columns: `attack,start,end` in RFC3339 UTC). The attack families it
generates:

| Attack | Tool | Signal it produces |
|--------|------|--------------------|
| `nmap_synscan` | `nmap -sS` over ports 1–10000 | high distinct destination ports/IPs in the rolling-60s window |
| `hping3_syn` | `hping3 -S` (non-flood, no spoof) | sustained SYN flow volume |
| `slowloris` | `slowhttptest` | many slow, long-lived L7 connections |
| `http_bruteforce` | `curl` credential loop | repeated L7 requests |

### Before running

1. A running `attacker` pod with `nmap`, `hping3`, `slowhttptest`, and `curl`.
2. A benign traffic generator running in the background, with the eBPF sensors capturing
   for **at least 5 minutes beforehand**.
3. Edit the six service/pod IPs at the top of the script (see the comments for the
   `kubectl` commands that print them).

```bash
bash run-attack.sh
```

Each attack writes its labelled interval to `labels.csv`. The script deliberately waits
(`SETTLE`) after each attack so the rolling-60s windows close before the label boundary,
and idles between attacks to keep classes separable. Keep the sensors capturing for
**≥3 minutes after** the last attack so its final window is fully formed.

The resulting `labels.csv` is the labelled timeline consumed by
[evaluate.py](../evaluate-and-verification/evaluate.py) and the other verification
scripts.
