"""
Streamlit admin panel.

Run:
    streamlit run admin_app.py

Pages
-----
  📹 Live        — camera feed + start/stop/restart stream controls
  👤 Users       — list registered users + delete
  ➕ Add User    — register new user with face photo
  📋 Sessions    — normalized event log
"""
import datetime
import io
import os
import signal
import subprocess
import time

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from db import (User, UserFeature, UserSession, Recording, PersonEvent,
                bytes_to_emb, emb_to_bytes, get_session)
from embeddings import FaceEmbedder
from post_analyser import PostAnalyser

st.set_page_config(page_title="People Recognition — Admin", layout="wide")

PID_FILE          = "stream.pid"
ANALYSER_PID_FILE = "analyser.pid"
PREVIEW_PATH      = "stream_preview.jpg"
STREAM_SCRIPT   = os.path.join(os.path.dirname(__file__), "stream.py")
ANALYSER_SCRIPT = os.path.join(os.path.dirname(__file__), "post_analyser.py")


# ── cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading face embedder…")
def get_embedder() -> FaceEmbedder:
    return FaceEmbedder()


def _get_analyser() -> PostAnalyser:
    with get_session() as db:
        rows = db.query(UserFeature).all()
        db_embeddings = [(r.user_id, bytes_to_emb(r.embedding)) for r in rows]
    return PostAnalyser(get_embedder(), db_embeddings)


def reanalyse_event(event_id: int) -> str:
    """Re-identify a single PersonEvent synchronously."""
    with get_session() as db:
        ev  = db.get(PersonEvent, event_id)
        if not ev:
            return "Event not found"
        rec = db.get(Recording, ev.recording_id)
        if not rec or not rec.video_path or not os.path.exists(rec.video_path):
            return "Video not found"
        first_seen  = ev.first_seen
        last_seen   = ev.last_seen or datetime.datetime.now()
        scene_start = rec.start_time
        video_path  = rec.video_path
    analyser = _get_analyser()
    uid = analyser.analyse_event_sync(
        event_id, video_path, scene_start, first_seen, last_seen
    )
    return uid or "Unknown"


# ── helpers ───────────────────────────────────────────────────────────────────

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_users() -> list[dict]:
    with get_session() as db:
        rows = db.query(User).order_by(User.created_at.desc()).all()
        return [{"id": u.id, "name": u.name, "created_at": u.created_at} for u in rows]


def load_recordings(limit: int = 50) -> list[dict]:
    with get_session() as db:
        rows = db.query(Recording).order_by(Recording.start_time.desc()).limit(limit).all()
        return [
            {"id": r.id, "start_time": r.start_time,
             "end_time": r.end_time, "video_path": r.video_path}
            for r in rows
        ]


def load_events(recording_id: int) -> list[dict]:
    with get_session() as db:
        rows = (
            db.query(PersonEvent)
            .filter_by(recording_id=recording_id)
            .order_by(PersonEvent.first_seen)
            .all()
        )
        return [
            {"id": e.id, "user_id": e.user_id, "track_id": e.track_id,
             "first_seen": e.first_seen, "last_seen": e.last_seen,
             "snapshot_path": e.snapshot_path, "recording_id": e.recording_id}
            for e in rows
        ]


def user_embedding_count(user_id: str) -> int:
    with get_session() as db:
        return db.query(UserFeature).filter_by(user_id=user_id).count()


@st.cache_data(show_spinner=False)
def extract_thumbnail(video_path: str, _mtime: float) -> bytes | None:
    """Return JPEG bytes of the middle frame; cached by path+mtime."""
    try:
        container = av.open(video_path)
        vid = container.streams.video[0]
        total = vid.frames or 0
        target = max(0, (total // 2) - 1)
        img = None
        for i, frame in enumerate(container.decode(vid)):
            img = frame.to_ndarray(format="bgr24")
            if i >= target:
                break
        container.close()
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None
    except Exception:
        return None


# ── stream process helpers ────────────────────────────────────────────────────

def _stream_pid() -> int | None:
    if not os.path.exists(PID_FILE):
        return None
    try:
        pid = int(open(PID_FILE).read().strip())
        os.kill(pid, 0)   # raises if process is dead
        return pid
    except (ValueError, OSError):
        return None


def _stream_running() -> bool:
    return _stream_pid() is not None


def _start_stream() -> None:
    import sys
    subprocess.Popen(
        [sys.executable, STREAM_SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_stream() -> None:
    pid = _stream_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _restart_stream() -> None:
    _stop_stream()
    time.sleep(1.5)
    _start_stream()


# ── Module B (analyser) process helpers ───────────────────────────────────────

def _analyser_pid() -> int | None:
    if not os.path.exists(ANALYSER_PID_FILE):
        return None
    try:
        pid = int(open(ANALYSER_PID_FILE).read().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        return None


def _analyser_running() -> bool:
    return _analyser_pid() is not None


def _start_analyser() -> None:
    import sys
    subprocess.Popen(
        [sys.executable, ANALYSER_SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_analyser() -> None:
    pid = _analyser_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _restart_analyser() -> None:
    _stop_analyser()
    time.sleep(1.5)
    _start_analyser()


# ── sidebar nav ───────────────────────────────────────────────────────────────

st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["📹 Live", "👤 Users", "➕ Add User", "📋 Sessions"],
                        label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live
# ══════════════════════════════════════════════════════════════════════════════
if page == "📹 Live":
    from streamlit_autorefresh import st_autorefresh   # type: ignore
    st_autorefresh(interval=500, limit=None, key="live_refresh")

    st.title("Live Camera")

    # ── Module A: recorder controls ───────────────────────────────────────────
    rec_running = _stream_running()
    rec_color   = "🟢" if rec_running else "🔴"
    st.markdown(f"**Module A — Recorder:** {rec_color} "
                f"{'Running' if rec_running else 'Stopped'}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start", disabled=rec_running, key="rec_start", width="stretch"):
            _start_stream()
            st.success("Recorder starting…")
            time.sleep(0.5)
            st.rerun()
    with c2:
        if st.button("⏹ Stop", disabled=not rec_running, key="rec_stop", width="stretch"):
            _stop_stream()
            st.success("Recorder stopping…")
            time.sleep(0.5)
            st.rerun()
    with c3:
        if st.button("🔄 Restart", key="rec_restart", width="stretch"):
            _restart_stream()
            st.info("Restarting recorder…")
            time.sleep(0.5)
            st.rerun()

    # ── Module B: analyser controls ───────────────────────────────────────────
    ana_running = _analyser_running()
    ana_color   = "🟢" if ana_running else "🔴"
    st.markdown(f"**Module B — Analyser:** {ana_color} "
                f"{'Running' if ana_running else 'Stopped'}")

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("▶ Start", disabled=ana_running, key="ana_start", width="stretch"):
            _start_analyser()
            st.success("Analyser starting…")
            time.sleep(0.5)
            st.rerun()
    with a2:
        if st.button("⏹ Stop", disabled=not ana_running, key="ana_stop", width="stretch"):
            _stop_analyser()
            st.success("Analyser stopping…")
            time.sleep(0.5)
            st.rerun()
    with a3:
        if st.button("🔄 Restart", key="ana_restart", width="stretch"):
            _restart_analyser()
            st.info("Restarting analyser…")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # ── preview image ─────────────────────────────────────────────────────────
    if os.path.exists(PREVIEW_PATH):
        mtime = os.path.getmtime(PREVIEW_PATH)
        age   = time.time() - mtime
        if age < 10:
            st.image(PREVIEW_PATH, width="stretch")
            st.caption(f"Last frame: {age:.1f}s ago")
        else:
            st.warning(f"Preview is stale ({age:.0f}s old) — stream may have stopped.")
    else:
        st.info("No preview available. Start the stream to see the camera feed.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Users
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Users":
    st.title("Registered Users")

    users = load_users()
    if not users:
        st.info("No users registered yet. Go to **➕ Add User** to register someone.")
    else:
        for u in users:
            with st.expander(f"**{u['name']}** — `{u['id']}`", expanded=False):
                emb_count = user_embedding_count(u["id"])
                ts = u["created_at"].strftime("%Y-%m-%d %H:%M") if u["created_at"] else "—"
                st.caption(f"Registered {ts} · {emb_count} face embedding(s)")

                # ── add more angles ───────────────────────────────────────────
                st.markdown("**Add more face angles** — improves recognition from different views")
                more_photo = st.camera_input("Take photo", key=f"cam_{u['id']}")

                col_add, col_del = st.columns([3, 1])
                with col_add:
                    if st.button("➕ Save this angle", key=f"add_{u['id']}", width="stretch"):
                        if more_photo is None:
                            st.warning("Take a photo first.")
                        else:
                            pil = Image.open(io.BytesIO(more_photo.getvalue()))
                            bgr = pil_to_bgr(pil)
                            with st.spinner("Extracting embedding…"):
                                emb = get_embedder().get_embedding(bgr)
                            if emb is None:
                                st.error("No face detected — use a clearer frontal shot.")
                            else:
                                # Diversity check against existing embeddings
                                with get_session() as db:
                                    existing_embs = [
                                        bytes_to_emb(r.embedding)
                                        for r in db.query(UserFeature)
                                                    .filter_by(user_id=u["id"]).all()
                                    ]
                                if existing_embs:
                                    max_sim = max(float(np.dot(emb, e)) for e in existing_embs)
                                    if max_sim > 0.82:
                                        st.warning(
                                            f"Too similar to an existing angle "
                                            f"({max_sim:.0%}). Try a different pose."
                                        )
                                    else:
                                        with get_session() as db:
                                            db.add(UserFeature(user_id=u["id"],
                                                               embedding=emb_to_bytes(emb)))
                                        st.success(f"Angle added ({emb_count + 1} total).")
                                        st.rerun()
                                else:
                                    with get_session() as db:
                                        db.add(UserFeature(user_id=u["id"],
                                                           embedding=emb_to_bytes(emb)))
                                    st.success("First angle saved.")
                                    st.rerun()
                with col_del:
                    if st.button("🗑 Delete user", key=f"del_{u['id']}", type="secondary"):
                        with get_session() as db:
                            row = db.get(User, u["id"])
                            if row:
                                db.delete(row)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Add User  —  Face ID-style guided scan
# ══════════════════════════════════════════════════════════════════════════════
elif page == "➕ Add User":

    # ── guided scan steps (label, arrow hint) ─────────────────────────────────
    _SCAN_STEPS = [
        ("Look straight at the camera",  ""),
        ("Turn your head slightly right", "→"),
        ("Turn your head slightly left",  "←"),
        ("Tilt your head slightly up",    "↑"),
        ("Tilt your head slightly down",  "↓"),
        ("Tilt slightly — any angle",     "↗"),
    ]
    _TOTAL = len(_SCAN_STEPS)
    _DIVERSITY_THRESHOLD = 0.82   # reject if too similar to an existing capture

    # ── session-state helpers ──────────────────────────────────────────────────
    def _fid_init():
        defaults = dict(fid_uid="", fid_name="", fid_step=0,
                        fid_embeddings=[], fid_done=False, fid_warn="")
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _fid_reset():
        for k in ("fid_uid","fid_name","fid_step","fid_embeddings","fid_done","fid_warn"):
            st.session_state.pop(k, None)

    _fid_init()

    # ── oval CSS ──────────────────────────────────────────────────────────────
    def _oval_html(step: int, arrow: str, label: str) -> str:
        pct   = int(step / _TOTAL * 100)
        color = "#34c759" if step > 0 else "#007aff"
        ring  = " ".join(
            '<span style="font-size:18px">●</span>' if i < step
            else ('<span style="font-size:18px;color:#007aff">●</span>' if i == step
                  else '<span style="font-size:18px;color:#3a3a3c">○</span>')
            for i in range(_TOTAL)
        )
        return f"""
        <div style="text-align:center;padding:12px 0 4px">
          <div style="display:inline-block;position:relative;
                      width:150px;height:190px;margin-bottom:8px">
            <svg width="150" height="190" style="position:absolute;top:0;left:0">
              <ellipse cx="75" cy="95" rx="70" ry="88"
                       fill="none" stroke="#3a3a3c" stroke-width="3"/>
              <ellipse cx="75" cy="95" rx="70" ry="88"
                       fill="none" stroke="{color}" stroke-width="3"
                       stroke-dasharray="{int(3.14159*(70+88)*2*pct/100)},9999"
                       stroke-linecap="round"
                       transform="rotate(-90 75 95)"/>
            </svg>
            <div style="position:absolute;top:50%;left:50%;
                        transform:translate(-50%,-50%);
                        font-size:52px;line-height:1">{arrow or "◎"}</div>
          </div>
          <p style="font-size:1.05em;font-weight:600;margin:4px 0">{label}</p>
          <div style="margin:6px 0">{ring}</div>
          <p style="color:#8e8e93;font-size:.9em;margin:0">
            {step} of {_TOTAL} scanned
          </p>
        </div>
        """

    # ── render ─────────────────────────────────────────────────────────────────
    if st.session_state.fid_done:
        # ── success screen ────────────────────────────────────────────────────
        n = len(st.session_state.fid_embeddings)
        st.markdown(f"""
        <div style="text-align:center;padding:32px 0">
          <div style="font-size:72px">✅</div>
          <h2 style="margin:8px 0">Face ID Setup Complete</h2>
          <p style="color:#8e8e93;font-size:1.05em">
            <b>{st.session_state.fid_name}</b> &nbsp;·&nbsp;
            <code>{st.session_state.fid_uid}</code><br>
            {n} face angles saved — recognition is now active
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Register Another Person", type="primary"):
            _fid_reset()
            st.rerun()

    elif not st.session_state.fid_uid:
        # ── step 0: enter name + ID ───────────────────────────────────────────
        st.title("Face ID Setup")
        st.caption("We'll guide you through 6 head positions — like Apple Face ID.")
        st.divider()
        with st.form("fid_info_form"):
            uid  = st.text_input("User ID (unique)", placeholder="emp_001")
            name = st.text_input("Full Name",        placeholder="John Doe")
            if st.form_submit_button("Start Face Scan →", type="primary"):
                if not uid.strip() or not name.strip():
                    st.error("Both fields are required.")
                else:
                    with get_session() as db:
                        exists = db.get(User, uid.strip())
                    if exists:
                        st.error(f"ID `{uid.strip()}` is already registered. "
                                 f"Go to 👤 Users to add more angles.")
                    else:
                        st.session_state.fid_uid  = uid.strip()
                        st.session_state.fid_name = name.strip()
                        st.rerun()

    else:
        # ── scanning steps ────────────────────────────────────────────────────
        step        = st.session_state.fid_step
        label, arrow = _SCAN_STEPS[step]

        st.title("Face ID Setup")
        st.caption(f"Setting up **{st.session_state.fid_name}** · "
                   f"`{st.session_state.fid_uid}`")

        col_oval, col_cam = st.columns([1, 1])

        with col_oval:
            st.markdown(_oval_html(step, arrow, label), unsafe_allow_html=True)
            if st.session_state.fid_warn:
                st.warning(st.session_state.fid_warn)
                st.session_state.fid_warn = ""

        with col_cam:
            photo = st.camera_input("", key=f"fid_cam_{step}",
                                    label_visibility="collapsed")

            if photo is not None:
                pil = Image.open(io.BytesIO(photo.getvalue()))
                bgr = pil_to_bgr(pil)

                with st.spinner("Analysing face…"):
                    emb = get_embedder().get_embedding(bgr)

                if emb is None:
                    st.error("No face detected — centre your face in the oval and retry.")
                else:
                    # Diversity gate
                    existing = st.session_state.fid_embeddings
                    if existing:
                        max_sim = max(float(np.dot(emb, e)) for e in existing)
                        if max_sim > _DIVERSITY_THRESHOLD:
                            st.session_state.fid_warn = (
                                f"Angle too similar to a previous capture "
                                f"(similarity {max_sim:.0%}). "
                                f"Please move your head more."
                            )
                            st.rerun()

                    st.session_state.fid_embeddings.append(emb)
                    st.session_state.fid_step += 1

                    if st.session_state.fid_step >= _TOTAL:
                        # Persist everything to DB in one transaction
                        try:
                            with get_session() as db:
                                db.add(User(
                                    id         = st.session_state.fid_uid,
                                    name       = st.session_state.fid_name,
                                    created_at = datetime.datetime.utcnow(),
                                ))
                                for e in st.session_state.fid_embeddings:
                                    db.add(UserFeature(
                                        user_id   = st.session_state.fid_uid,
                                        embedding = emb_to_bytes(e),
                                    ))
                            st.session_state.fid_done = True
                        except Exception as ex:
                            st.error(f"Database error: {ex}")
                    st.rerun()

        if st.button("✕ Cancel setup"):
            _fid_reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Access Log
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Sessions":
    st.title("Access Log")

    with get_session() as db:
        users_map = {u.id: u.name for u in db.query(User).all()}

    # ── toolbar ───────────────────────────────────────────────────────────────
    col_r, col_clr = st.columns([3, 1])
    with col_r:
        if st.button("🔄 Re-analyze all unknowns", width="stretch"):
            with get_session() as db:
                unknown_events = (
                    db.query(PersonEvent)
                    .filter(PersonEvent.user_id == None)
                    .all()
                )
                targets = []
                for ev in unknown_events:
                    rec = db.get(Recording, ev.recording_id)
                    if rec and rec.video_path and os.path.exists(rec.video_path):
                        targets.append({
                            "event_id":   ev.id,
                            "video_path": rec.video_path,
                            "scene_start": rec.start_time,
                            "first_seen": ev.first_seen,
                            "last_seen":  ev.last_seen or datetime.datetime.now(),
                        })
            if not targets:
                st.info("No unknown events with video to re-analyze.")
            else:
                bar      = st.progress(0, text=f"0/{len(targets)}…")
                analyser = _get_analyser()
                for idx, t in enumerate(targets):
                    analyser.analyse_event_sync(
                        t["event_id"], t["video_path"], t["scene_start"],
                        t["first_seen"], t["last_seen"],
                    )
                    bar.progress((idx + 1) / len(targets),
                                 text=f"{idx+1}/{len(targets)}…")
                st.success(f"Done — re-analyzed {len(targets)} event(s).")
                st.rerun()
    with col_clr:
        if st.button("🗑 Clear all", type="secondary", width="stretch"):
            with get_session() as db:
                db.query(PersonEvent).delete()
                db.query(Recording).delete()
            st.success("All recordings cleared.")
            st.rerun()

    show_unknown = st.checkbox("Show unidentified persons", value=True)

    recordings = load_recordings()
    if not recordings:
        st.info("No recordings yet. Start the stream and walk in front of the camera.")
    else:
        for rec in recordings:
            events    = load_events(rec["id"])
            n_total   = len(events)
            n_unknown = sum(1 for e in events if not e["user_id"])

            if not show_unknown and n_unknown == n_total:
                continue

            # ── recording header ──────────────────────────────────────────────
            dur_str = "ongoing"
            if rec["start_time"] and rec["end_time"]:
                sec     = int((rec["end_time"] - rec["start_time"]).total_seconds())
                dur_str = f"{sec // 60}m {sec % 60}s"
            start_str = rec["start_time"].strftime("%Y-%m-%d %H:%M:%S") if rec["start_time"] else "—"

            names = [users_map.get(e["user_id"], "Unknown") if e["user_id"] else "Unknown"
                     for e in events]
            people_str = ", ".join(dict.fromkeys(names)) or "—"   # unique, ordered

            with st.expander(
                f"📹 {start_str}  |  {dur_str}  |  {n_total} person(s): {people_str}",
                expanded=False,
            ):
                # ── person cards (one per PersonEvent) ─────────────────────────
                n_cols = min(n_total, 4) or 1
                cols   = st.columns(n_cols)

                for i, ev in enumerate(events):
                    if not show_unknown and not ev["user_id"]:
                        continue
                    name  = users_map.get(ev["user_id"], "Unknown") if ev["user_id"] else "Unknown"
                    label = f"{name}\n`{ev['user_id']}`" if ev["user_id"] else "Unknown"
                    identified = bool(ev["user_id"])

                    t_in  = ev["first_seen"].strftime("%H:%M:%S") if ev["first_seen"] else "—"
                    t_out = ev["last_seen"].strftime("%H:%M:%S")  if ev["last_seen"]  else "ongoing"
                    if ev["first_seen"] and ev["last_seen"]:
                        dur = int((ev["last_seen"] - ev["first_seen"]).total_seconds())
                        t_dur = f"{dur}s"
                    else:
                        t_dur = "—"

                    with cols[i % n_cols]:
                        # Snapshot photo
                        snap = ev["snapshot_path"]
                        if snap and os.path.exists(snap):
                            st.image(snap, width="stretch")
                        else:
                            st.markdown("📷 *no snapshot*")

                        # Identity badge
                        if identified:
                            st.success(f"**{name}**\n\n`{ev['user_id']}`")
                        else:
                            st.warning("**Unknown**")

                        st.caption(f"Track #{ev['track_id']}  |  {t_in} – {t_out}  ({t_dur})")

                        # Re-analyze button for unknown events
                        if not identified:
                            vpath = rec["video_path"]
                            if vpath and os.path.exists(vpath):
                                if st.button("🔄 Re-analyze",
                                             key=f"ra_{ev['id']}",
                                             width="stretch"):
                                    with st.spinner("Identifying…"):
                                        result = reanalyse_event(ev["id"])
                                    st.success(f"→ {result}")
                                    st.rerun()

                st.divider()

                # ── shared video player ────────────────────────────────────────
                vpath = rec["video_path"]
                if vpath and os.path.exists(vpath):
                    st.markdown("**Scene recording**")
                    with open(vpath, "rb") as _f:
                        st.video(_f.read())
                    st.caption(f"`{os.path.basename(vpath)}`")
                else:
                    st.caption("Video file not found.")
