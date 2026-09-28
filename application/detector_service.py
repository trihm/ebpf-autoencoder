#!/usr/bin/env python3
"""
detector_service.py — detector live + dashboard near realtime.

Luong: tail NDJSON -> StreamAggregator (cua so 5s) -> dual-AE score + fusion
-> gom thanh incident -> day len browser qua SSE.

Chay:
  uvicorn detector_service:app --host 0.0.0.0 --port 8000

Bien moi truong (hoac sua DEFAULTS ben duoi):
  HUBBLE_LOG, TETRAGON_LOG, MODEL_DIR, WINDOW, ROLLING, GRACE, FUSION_MODE,
  INCIDENT_GAP, DB_PATH
"""
import json
import os
import queue
import sqlite3
import threading
import time
from collections import deque

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from detector_core import (
    NETWORK_FEATURES, PROCESS_FEATURES, Branch, fuse_rows, src_ip_of,
)
from stream_features import (
    LogTailer, StreamAggregator, parse_hubble_obj, parse_tetragon_obj,
)

HUBBLE_LOG = os.environ.get("HUBBLE_LOG", "/tmp/ebpf-logs/cilium-network.json")
TETRAGON_LOG = os.environ.get("TETRAGON_LOG", "/tmp/ebpf-logs/tetragon-events.json")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/ebpf-logs/models")
WINDOW = int(os.environ.get("WINDOW", "5"))
ROLLING = int(os.environ.get("ROLLING", "60"))
GRACE = float(os.environ.get("GRACE", "2.0"))
FUSION_MODE = os.environ.get("FUSION_MODE", "max")
INCIDENT_GAP = float(os.environ.get("INCIDENT_GAP", "15"))
DB_PATH = os.environ.get("DB_PATH", "/tmp/ebpf-logs/alerts.db")
HISTORY_N = 720          # ~1 gio cua so 5s giu trong RAM de ve timeline

app = FastAPI(title="eBPF dual-AE detector")

state = {
    "history": deque(maxlen=HISTORY_N),   # 1 diem / cua so
    "incidents": {},                      # src_ip -> incident dang mo
    "closed": deque(maxlen=200),
    "stats": {"windows": 0, "alerts": 0, "net_events": 0, "proc_events": 0,
              "lag": None, "last_net_event": None, "last_proc_event": None},
    "lock": threading.Lock(),
    "subs": [],                           # list[queue.Queue]
}


# --------------------------------------------------------------------------
# Luu tru
# --------------------------------------------------------------------------

def db_init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS incident(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        src_ip TEXT, start_ts REAL, end_ts REAL, n_windows INTEGER,
        max_score REAL, trigger TEXT, branch TEXT, top_features TEXT,
        sample_flow TEXT)""")
    con.commit()
    return con


DB = db_init()


def db_save(inc):
    with state["lock"]:
        DB.execute(
            "INSERT INTO incident(src_ip,start_ts,end_ts,n_windows,max_score,"
            "trigger,branch,top_features,sample_flow) VALUES (?,?,?,?,?,?,?,?,?)",
            (inc["src_ip"], inc["start_ts"], inc["end_ts"], inc["n_windows"],
             inc["max_score"], inc["trigger"], inc["branch"],
             json.dumps(inc["top"], ensure_ascii=False), inc["sample_flow"]),
        )
        DB.commit()


# --------------------------------------------------------------------------
# Phat su kien cho browser
# --------------------------------------------------------------------------

def broadcast(kind, payload):
    msg = json.dumps({"type": kind, "data": payload}, ensure_ascii=False)
    with state["lock"]:
        subs = list(state["subs"])
    for q in subs:
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


# --------------------------------------------------------------------------
# Gom alert thanh incident (theo source IP)
# --------------------------------------------------------------------------

def update_incidents(win, alerts, now_wall):
    """alerts: list dong da vuot nguong trong cua so nay."""
    touched = set()
    for row in alerts:
        ip = src_ip_of(row["flow_key"])
        touched.add(ip)
        inc = state["incidents"].get(ip)
        if inc is None:
            inc = {
                "src_ip": ip, "start_ts": win, "end_ts": win + WINDOW,
                "n_windows": 0, "max_score": 0.0, "trigger": row["trigger"],
                "branch": row["branch"], "top": row["top"],
                "sample_flow": row["flow_key"], "first_seen_wall": now_wall,
            }
            state["incidents"][ip] = inc
        inc["end_ts"] = win + WINDOW
        inc["n_windows"] += 1
        if row["score"] > inc["max_score"]:
            inc.update(max_score=row["score"], trigger=row["trigger"],
                       branch=row["branch"], top=row["top"],
                       sample_flow=row["flow_key"])

    # dong incident im lang qua lau
    for ip in list(state["incidents"]):
        inc = state["incidents"][ip]
        if ip not in touched and (win - inc["end_ts"]) > INCIDENT_GAP:
            del state["incidents"][ip]
            state["closed"].appendleft(inc)
            db_save(inc)
            broadcast("incident_closed", inc)

    for ip in touched:
        broadcast("incident", state["incidents"][ip])


# --------------------------------------------------------------------------
# Vong lap nen
# --------------------------------------------------------------------------

def worker():
    net_branch = Branch("network",
                        f"{MODEL_DIR}/net_model.pt",
                        f"{MODEL_DIR}/net_scaler.pkl",
                        f"{MODEL_DIR}/net_thr.txt",
                        NETWORK_FEATURES)
    proc_branch = Branch("process",
                         f"{MODEL_DIR}/proc_model.pt",
                         f"{MODEL_DIR}/proc_scaler.pkl",
                         f"{MODEL_DIR}/proc_thr.txt",
                         PROCESS_FEATURES)
    print(f"[detector] nguong net={net_branch.threshold:.4f} "
          f"proc={proc_branch.threshold:.4f} "
          f"winsor_net={len(net_branch.caps)} winsor_proc={len(proc_branch.caps)}")
    if not net_branch.caps or not proc_branch.caps:
        print("[detector] CANH BAO: khong tim thay *_winsor.npz — so lieu se lech "
              "so voi bao cao eval. Kiem tra lai MODEL_DIR.")

    agg = StreamAggregator(window=WINDOW, rolling=ROLLING, grace=GRACE)
    t_net = LogTailer(HUBBLE_LOG)
    t_proc = LogTailer(TETRAGON_LOG)

    while True:
        n_net = n_proc = 0
        for obj in t_net.read_objects():
            r = parse_hubble_obj(obj)
            if r:
                agg.add_network(r)
                n_net += 1
                state["stats"]["last_net_event"] = r["t"]
        for obj in t_proc.read_objects():
            r = parse_tetragon_obj(obj)
            if r:
                agg.add_process(r)
                n_proc += 1
                state["stats"]["last_proc_event"] = r["t"]
        state["stats"]["net_events"] += n_net
        state["stats"]["proc_events"] += n_proc

        for win, net_rows, proc_rows in agg.pop_ready():
            t_score = time.time()
            rows = fuse_rows(net_rows, proc_rows, net_branch, proc_branch,
                             mode=FUSION_MODE)
            alerts = [r for r in rows if r["alert"]]
            norms_net = [r["norm_net"] for r in rows if r["norm_net"] is not None]
            norms_proc = [r["norm_proc"] for r in rows if r["norm_proc"] is not None]
            point = {
                "window_start": win,
                "max_norm_net": round(max(norms_net), 4) if norms_net else 0.0,
                "max_norm_proc": round(max(norms_proc), 4) if norms_proc else 0.0,
                "n_flows": len(rows),
                "n_alerts": len(alerts),
                "branch_mix": {
                    b: sum(1 for r in rows if r["branch"] == b)
                    for b in ("both", "network_only", "process_only")
                },
                "lag": round(t_score - (win + WINDOW), 2),
                "infer_ms": round((time.time() - t_score) * 1000, 1),
            }
            state["history"].append(point)
            state["stats"]["windows"] += 1
            state["stats"]["alerts"] += len(alerts)
            state["stats"]["lag"] = point["lag"]
            broadcast("window", point)
            update_incidents(win, alerts, t_score)

        time.sleep(0.25)


@app.on_event("startup")
def start_worker():
    th = threading.Thread(target=worker, daemon=True, name="detector")
    th.start()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "dashboard.html"))


@app.get("/api/history")
def history():
    return {
        "window": WINDOW,
        "history": list(state["history"]),
        "open_incidents": list(state["incidents"].values()),
        "closed_incidents": list(state["closed"])[:50],
        "stats": state["stats"],
    }


@app.get("/api/health")
def health():
    now = time.time()
    s = state["stats"]
    def age(ts):
        return None if ts is None else round(now - ts, 1)
    return {
        "net_event_age_s": age(s["last_net_event"]),
        "proc_event_age_s": age(s["last_proc_event"]),
        "windows": s["windows"], "alerts": s["alerts"], "lag_s": s["lag"],
    }


@app.get("/events")
def events():
    q = queue.Queue(maxsize=1000)
    with state["lock"]:
        state["subs"].append(q)

    def gen():
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with state["lock"]:
                if q in state["subs"]:
                    state["subs"].remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
