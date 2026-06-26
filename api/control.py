"""Module process management: start/stop/restart, watchdog, desired-state persistence."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from .auth import require_auth
from .deps import (ANALYSER_PID, ANALYSER_SCRIPT, LOGS_DIR, PREVIEW_PATH,
                   PID_FILE, STATE_FILE, STREAM_SCRIPT)
from log_setup import get_logger

log = get_logger("server", "server.log")
router = APIRouter()


# ── process helpers ───────────────────────────────────────────────────────────

def read_pid(path: str) -> Optional[int]:
    try:
        pid = int(open(path).read().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, OSError):
        return None


def proc_running(pid_file: str) -> bool:
    return read_pid(pid_file) is not None


def start_proc(script: str) -> None:
    subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_proc(pid_file: str) -> None:
    pid = read_pid(pid_file)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def restart_proc(pid_file: str, script: str) -> None:
    stop_proc(pid_file)
    time.sleep(1.5)
    start_proc(script)


# ── desired-state persistence ─────────────────────────────────────────────────

def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"recorder": False, "analyser": False}


def save_state(**kwargs) -> None:
    state = load_state()
    state.update(kwargs)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)


# ── watchdog ──────────────────────────────────────────────────────────────────

def _watchdog_loop() -> None:
    while True:
        try:
            s = load_state()
            if s.get("recorder") and not proc_running(PID_FILE):
                log.info("watchdog: restarting recorder")
                start_proc(STREAM_SCRIPT)
                # Stagger: give the recorder 10 s to load before the analyser
                # starts its own heavy models (buffalo_l + YOLO).  Prevents both
                # processes from spiking RAM simultaneously on pod restart.
                if s.get("analyser") and not proc_running(ANALYSER_PID):
                    time.sleep(10)
            if s.get("analyser") and not proc_running(ANALYSER_PID):
                log.info("watchdog: restarting analyser")
                start_proc(ANALYSER_SCRIPT)
        except Exception as e:
            log.warning("watchdog error: %s", e)
        time.sleep(15)


def watchdog_start() -> None:
    threading.Thread(target=_watchdog_loop, daemon=True, name="watchdog").start()


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/status")
def api_status(_=Depends(require_auth)):
    age = None
    if os.path.exists(PREVIEW_PATH):
        age = round(time.time() - os.path.getmtime(PREVIEW_PATH), 1)
    return {
        "recorder":    proc_running(PID_FILE),
        "analyser":    proc_running(ANALYSER_PID),
        "preview_age": age,
    }


@router.post("/control/{module}/{action}")
def control(module: str, action: str, _=Depends(require_auth)):
    _map = {
        "recorder": (PID_FILE, STREAM_SCRIPT,   "recorder"),
        "analyser": (ANALYSER_PID, ANALYSER_SCRIPT, "analyser"),
    }
    if module not in _map:
        raise HTTPException(400, "unknown module")
    pid_file, script, key = _map[module]

    if action == "start":
        if not proc_running(pid_file):
            start_proc(script)
        save_state(**{key: True})
    elif action == "stop":
        stop_proc(pid_file)
        save_state(**{key: False})
    elif action == "restart":
        restart_proc(pid_file, script)
        save_state(**{key: True})
    else:
        raise HTTPException(400, "unknown action")

    return {"ok": True}
