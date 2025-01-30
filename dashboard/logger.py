import datetime as dt
import json
import os
import streamlit as st
from pathlib import Path

LOG_DIR = "logs/"


def log(sid, action, dashboard_state, query_params=None):
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    time_txt = now.strftime("%Y-%m-%d %H:%M:%S")
    time_sec = int(now.strftime("%s"))
    x = {
        "session_id": sid,
        "time": time_txt,
        "time_sec": time_sec,
        "dashboard_state": dashboard_state,
        "action": action,
        "query_params": query_params,
    }
    log_path = os.path.join(LOG_DIR, sid + ".jsonl")
    with open(log_path, "a") as fs_out:
        print(json.dumps(x), file=fs_out)


def init():
    pass
