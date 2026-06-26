"""Calendar view and daily-report generation with combined video.

Edge cases handled:
- No recordings for the day → text-only report, no video
- Some video files deleted → skipped in concat, events still included
- Today vs past day → end boundary is now() vs 23:59:59
- Concurrent generation for same date → per-date lock prevents double work
- ffmpeg -c copy fails (mixed resolutions) → retry with scale + re-encode
- ffmpeg not available → video_error set, text report still returned
- Server restart → job not in memory but file may exist on disk
"""
from __future__ import annotations

import datetime
import os
import subprocess
import tempfile
import threading
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from .auth import require_auth
from .deps import BASE_DATA
from db import PersonEvent, Recording, User, get_session

router = APIRouter()

REPORTS_DIR = os.path.join(BASE_DATA, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# In-memory job registry  {date_str → job_dict}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Per-date generation lock — prevents two concurrent jobs for the same date
_gen_locks: dict[str, threading.Lock] = {}
_gen_locks_mtx = threading.Lock()


def _gen_lock(date_str: str) -> threading.Lock:
    with _gen_locks_mtx:
        if date_str not in _gen_locks:
            _gen_locks[date_str] = threading.Lock()
        return _gen_locks[date_str]


# ── calendar ──────────────────────────────────────────────────────────────────

@router.get("/calendar")
def get_calendar(
    month: str = Query(...),
    _: str = Depends(require_auth),
):
    """
    Per-day summary for a given month (YYYY-MM).
    Returns [{date, recordings, persons, unknown}].
    """
    try:
        if len(month) != 7 or month[4] != "-":
            raise ValueError
        year, mon = int(month[:4]), int(month[5:7])
        if not (1 <= mon <= 12):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "month must be YYYY-MM")

    start_dt = datetime.datetime(year, mon, 1)
    end_dt = (
        datetime.datetime(year + 1, 1, 1) if mon == 12
        else datetime.datetime(year, mon + 1, 1)
    )

    with get_session() as db:
        recs = (
            db.query(Recording)
            .filter(Recording.start_time >= start_dt, Recording.start_time < end_dt)
            .all()
        )
        umap = {u.id: u.name for u in db.query(User).all()}

        by_day: dict[str, dict] = {}
        for rec in recs:
            day = rec.start_time.strftime("%Y-%m-%d")
            if day not in by_day:
                by_day[day] = {"date": day, "recordings": 0, "known": set(), "unknown": 0}
            by_day[day]["recordings"] += 1
            for e in rec.events:
                if e.user_id:
                    by_day[day]["known"].add(umap.get(e.user_id, e.user_id))
                else:
                    by_day[day]["unknown"] += 1

    return [
        {
            "date":       day,
            "recordings": v["recordings"],
            "persons":    sorted(v["known"]),
            "unknown":    v["unknown"],
        }
        for day, v in sorted(by_day.items())
    ]


# ── report generation ─────────────────────────────────────────────────────────

@router.post("/reports/generate")
def generate_report(
    background_tasks: BackgroundTasks,
    date: str = Query(...),
    tz_offset: int = Query(0, ge=-720, le=840),
    _: str = Depends(require_auth),
):
    """
    Start (or re-start) background generation of a daily report.
    date     — YYYY-MM-DD in the client's local timezone
    tz_offset — client's offset in minutes east of UTC (from new Date().getTimezoneOffset() * -1)

    Today  → period 00:00 .. current time
    Past   → period 00:00 .. 23:59:59
    Future → period 00:00 .. 23:59:59 (will be empty)
    """
    try:
        if len(date) != 10 or date[4] != "-" or date[7] != "-":
            raise ValueError
        local_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    tz = datetime.timezone(datetime.timedelta(minutes=tz_offset))
    local_midnight = datetime.datetime(
        local_date.year, local_date.month, local_date.day, tzinfo=tz
    )
    utc_start = local_midnight.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    local_now = datetime.datetime.now(tz)
    if local_date >= local_now.date():
        # Today (or future): end at current UTC moment
        utc_end = datetime.datetime.utcnow()
    else:
        # Past day: include full 24 hours
        utc_end = utc_start + datetime.timedelta(hours=24, seconds=-1)

    job_id = date
    with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing and existing.get("status") == "generating":
            return {"job_id": job_id, **existing}
        _jobs[job_id] = {"status": "generating", "progress": "Queued…"}

    background_tasks.add_task(_run_report, job_id, utc_start, utc_end, local_date)
    return {"job_id": job_id, "status": "generating", "progress": "Queued…"}


@router.get("/reports/{job_id}")
def get_report(job_id: str, _: str = Depends(require_auth)):
    """Poll for report status. 404 if never generated."""
    with _jobs_lock:
        job = dict(_jobs.get(job_id) or {})
    if job:
        return job

    # After server restart the job dict is gone, but the video may still exist
    video_path = os.path.join(REPORTS_DIR, f"report_{job_id}.mp4")
    if os.path.isfile(video_path):
        return {
            "status":      "ready",
            "video_ready": True,
            "summary":     None,
            "note":        "Loaded from cache (server was restarted)",
        }

    raise HTTPException(404, "Report not found — click Generate")


@router.get("/reports/{job_id}/video")
def download_report_video(job_id: str, _: str = Depends(require_auth)):
    """Download the combined day video as an attachment."""
    video_path = os.path.join(REPORTS_DIR, f"report_{job_id}.mp4")
    if not os.path.isfile(video_path):
        raise HTTPException(404, "Video not ready or not available")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="report_{job_id}.mp4"'},
    )


# ── background worker ─────────────────────────────────────────────────────────

def _run_report(
    job_id: str,
    utc_start: datetime.datetime,
    utc_end: datetime.datetime,
    local_date: datetime.date,
) -> None:
    lock = _gen_lock(job_id)
    if not lock.acquire(blocking=False):
        return  # already running for this date
    try:
        _generate_report(job_id, utc_start, utc_end, local_date)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(
            "report %s failed: %s", job_id, exc, exc_info=True
        )
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(exc)}
    finally:
        lock.release()


def _generate_report(
    job_id: str,
    utc_start: datetime.datetime,
    utc_end: datetime.datetime,
    local_date: datetime.date,
) -> None:
    def progress(msg: str) -> None:
        with _jobs_lock:
            if _jobs.get(job_id, {}).get("status") == "generating":
                _jobs[job_id]["progress"] = msg

    progress("Loading recordings…")

    with get_session() as db:
        recs = (
            db.query(Recording)
            .filter(
                Recording.start_time >= utc_start,
                Recording.start_time <= utc_end,
            )
            .order_by(Recording.start_time)
            .all()
        )
        umap = {u.id: u.name for u in db.query(User).all()}

        events_out: list[dict] = []
        video_paths: list[str] = []

        for rec in recs:
            if rec.video_path and os.path.isfile(rec.video_path):
                video_paths.append(rec.video_path)
            for e in rec.events:
                events_out.append({
                    "id":              e.id,
                    "recording_id":    rec.id,
                    "rec_start":       rec.start_time.isoformat() if rec.start_time else None,
                    "user_id":         e.user_id,
                    "user_name":       umap.get(e.user_id) if e.user_id else None,
                    "first_seen":      e.first_seen.isoformat() if e.first_seen else None,
                    "last_seen":       e.last_seen.isoformat()  if e.last_seen  else None,
                    "snapshot_path":   e.snapshot_path,
                    "snapshot_exists": bool(
                        e.snapshot_path and os.path.isfile(e.snapshot_path)
                    ),
                })

    events_out.sort(key=lambda e: e["first_seen"] or "")

    known  = [e for e in events_out if e["user_id"]]
    unkn   = [e for e in events_out if not e["user_id"]]

    summary = {
        "date":            str(local_date),
        "period_start":    utc_start.isoformat(),
        "period_end":      utc_end.isoformat(),
        "recording_count": len(recs),
        "event_count":     len(events_out),
        "known_count":     len(known),
        "unknown_count":   len(unkn),
        "events":          events_out,
    }

    # ── video concatenation ───────────────────────────────────────────────────
    video_error: Optional[str] = None
    video_ready = False

    if not video_paths:
        video_error = "No video files found for this day"
    else:
        out_path = os.path.join(REPORTS_DIR, f"report_{job_id}.mp4")
        progress(f"Concatenating {len(video_paths)} recording(s)…")
        try:
            _concat_ffmpeg(video_paths, out_path)
            video_ready = True
        except FileNotFoundError:
            video_error = "ffmpeg not found — install ffmpeg to enable video reports"
        except subprocess.TimeoutExpired:
            video_error = "Video generation timed out (too many/large recordings)"
        except Exception as exc:
            video_error = f"Video generation failed: {exc}"

    with _jobs_lock:
        _jobs[job_id] = {
            "status":      "ready",
            "summary":     summary,
            "video_ready": video_ready,
            "video_error": video_error,
        }


def _concat_ffmpeg(paths: list[str], out_path: str) -> None:
    """
    Concatenate MP4s with ffmpeg.
    First attempt: stream copy (fast, lossless).
    If that fails (mixed resolutions/codecs): re-encode to 640p h264.
    """
    fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            for p in paths:
                escaped = p.replace("\\", "\\\\").replace("'", "\\'")
                f.write(f"file '{escaped}'\n")

        if os.path.exists(out_path):
            os.remove(out_path)

        base_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
        ]

        # Attempt 1: stream copy (no re-encode — fast)
        r = subprocess.run(
            base_cmd + ["-c", "copy", out_path],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            return

        # Attempt 2: re-encode to uniform 640p h264 (handles mixed resolutions)
        if os.path.exists(out_path):
            os.remove(out_path)
        r2 = subprocess.run(
            base_cmd + [
                "-vf", "scale=640:-2",
                "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                "-an",
                out_path,
            ],
            capture_output=True, text=True, timeout=600,
        )
        if r2.returncode != 0:
            raise RuntimeError(r2.stderr[-600:] if r2.stderr else "ffmpeg error")
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
