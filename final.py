import streamlit as st
import cv2
import math
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

st.set_page_config(page_title="Exam Cheating Detection", layout="wide")
st.title("🚨 Exam Cheating Detection System")

# ── Constants ──────────────────────────────────────────────────────────────
POSE_MODEL = "yolov8n-pose.pt"
OBJ_MODEL  = "yolov8n.pt"

FRAME_SKIP         = 5
PHONE_DETECT_GAP   = 3
CALIBRATION_FRAMES = 60
COPY_FRAMES        = 10
PHONE_FRAMES       = 6
ANGLE_THRESHOLD    = 20
PHONE_HAND_DIST    = 50
SEAT_DIST_THRESH   = 80
COOLDOWN_FRAMES    = 50
SLOW_FACTOR        = 0.3


# ── Helper functions ───────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag == 0:
        return 180
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def paper_point(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)//2, int(y2 - (y2-y1)*0.25))


def assign_seat(center):
    for sid, pos in st.session_state.seats.items():
        if dist(center, pos) < SEAT_DIST_THRESH:
            return sid
    new_id = st.session_state.next_seat_id
    st.session_state.seats[new_id] = center
    st.session_state.next_seat_id += 1
    return new_id


# ── Result renderer ────────────────────────────────────────────────────────
def render_results():
    """
    Draws the full results UI from session state.
    Runs at the TOP of every rerun so clicking any download button
    (which triggers a rerun) re-draws everything — screen never clears.
    """
    cheat_frequency = st.session_state.cheat_frequency
    events          = st.session_state.events
    seats           = st.session_state.seats
    out_path        = st.session_state.out_path
    fps             = st.session_state.video_fps

    # ── Video playback & download ──────────────────────────────────
    st.header("🎬 Annotated Video")
    if os.path.exists(out_path):
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(
                label="📥 Download Output Video",
                data=f,
                file_name="cheating_detection_output.mp4",
                mime="video/mp4",
                key="video_dl_btn"
            )
    else:
        st.error("❌ Output video not found")

    # ── Cheating frequency chart (from 29.py) ─────────────────────
    st.header("📊 Cheating Frequency Chart")

    all_student_ids = sorted(seats.keys())
    full_freq = {sid: cheat_frequency.get(sid, 0) for sid in all_student_ids}

    freq_df = (
        pd.DataFrame(full_freq.items(), columns=["Student ID", "Cheating Count"])
        .sort_values("Student ID")
    )
    freq_df["Student Label"] = freq_df["Student ID"].apply(lambda x: f"Student {x}")

    fig, ax = plt.subplots(figsize=(8, max(3, len(freq_df) * 0.5)))
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in freq_df["Cheating Count"]]
    bars = ax.barh(freq_df["Student Label"], freq_df["Cheating Count"], color=colors)

    ax.set_xlabel("Cheating Count")
    ax.set_title("Cheating Frequency per Student")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for bar, val in zip(bars, freq_df["Cheating Count"]):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9
        )

    ax.invert_yaxis()   # Student 1 at top
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── CSV report ────────────────────────────────────────────────
    if events:
        df = pd.DataFrame(events, columns=["Frame", "Type", "Student", "Target"])
        df["Student"] = df["Student"].apply(
            lambda x: f"Student {x}" if str(x).isdigit() else x
        )
        df["Target"] = df["Target"].apply(
            lambda x: f"Student {x}" if str(x).isdigit() else x
        )
        st.download_button(
            "📥 Download Cheating Report (CSV)",
            data=df.to_csv(index=False),
            file_name="exam_cheating_report.csv",
            mime="text/csv",
            key="csv_dl_btn"
        )
    else:
        st.success("✅ No cheating detected")


# ══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL ROUTING
# ══════════════════════════════════════════════════════════════════════════

# Always show uploader + process button (even after results are displayed)
uploaded_video = st.file_uploader(
    "Upload Exam Hall Video",
    type=["mp4", "avi", "mov"]
)
process_btn = st.button("▶️ Process Video")

# If results exist, render them below the uploader on every rerun
# (including reruns triggered by download-button clicks)
if st.session_state.get("processing_done"):
    render_results()

# Only proceed to processing if both uploader and button are active
if not (uploaded_video and process_btn):
    st.stop()

# ── Processing ─────────────────────────────────────────────────────────────
# Reset seat state for a fresh run
st.session_state.seats        = {}
st.session_state.next_seat_id = 1
st.session_state.processing_done = False

status = st.empty()
status.info("🔍 Processing video… Please wait")

tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_video.read())
tfile.flush()

cap          = cv2.VideoCapture(tfile.name)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 25

fourcc   = cv2.VideoWriter_fourcc(*'avc1')
out_path = tempfile.mktemp(suffix="_output.mp4")
out      = cv2.VideoWriter(out_path, fourcc, max(1, fps * SLOW_FACTOR), (480, 270))

pose_model = YOLO(POSE_MODEL)
obj_model  = YOLO(OBJ_MODEL)

frame_id      = 0
width_samples = []

DIST_THRESHOLD = 180
ROW_THRESHOLD  = 90

copy_counter     = {}
phone_counter    = {}
last_event_frame = {}

cheat_frequency = defaultdict(int)
events          = []
phones_last     = []

progress = st.progress(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (480, 270))

    # ── Pose estimation ──────────────────────────────────────────
    pose_res = pose_model.predict(frame, conf=0.5, verbose=False)[0]
    persons  = []

    if pose_res.keypoints is not None:
        for i in range(len(pose_res.keypoints.xy)):
            box  = pose_res.boxes.xyxy[i]
            kpts = pose_res.keypoints.xy[i]

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            nose, le, re = kpts[0], kpts[1], kpts[2]
            lw,   rw     = kpts[9], kpts[10]

            if nose[0] == 0 or le[0] == 0 or re[0] == 0:
                continue
            if dist(nose, (0, 0)) < 5:
                continue

            seat_id  = assign_seat((cx, cy))
            face_mid = ((le[0]+re[0])/2, (le[1]+re[1])/2)
            head_vec = (nose[0]-face_mid[0], nose[1]-face_mid[1])

            persons.append({
                "seat_id": seat_id,
                "box":     (x1, y1, x2, y2),
                "center":  (cx, cy),
                "nose":    nose,
                "head_vec":head_vec,
                "lw":      lw,
                "rw":      rw,
            })

            if frame_id < CALIBRATION_FRAMES:
                width_samples.append(x2 - x1)

    if frame_id == CALIBRATION_FRAMES and width_samples:
        avg            = sum(width_samples) / len(width_samples)
        DIST_THRESHOLD = int(avg * 1.5)
        ROW_THRESHOLD  = int(avg * 0.8)

    # ── Object detection (phones) ────────────────────────────────
    if frame_id % PHONE_DETECT_GAP == 0:
        obj_res = obj_model.predict(frame, conf=0.3, verbose=False)[0]
        phones  = []
        for box, cls in zip(obj_res.boxes.xyxy, obj_res.boxes.cls):
            if int(cls) == 67:
                x1, y1, x2, y2 = map(int, box)
                phones.append(((x1+x2)//2, (y1+y2)//2))
        phones_last = phones
    else:
        phones = phones_last

    # ── Phone detection ──────────────────────────────────────────
    for p in persons:
        for ph in phones:
            if ph[1] < p["nose"][1]:
                continue
            key       = f"phone-{p['seat_id']}"
            hand_dist = min(dist(ph, p["lw"]), dist(ph, p["rw"]))
            if hand_dist < PHONE_HAND_DIST:
                phone_counter[key] = phone_counter.get(key, 0) + 1
            else:
                phone_counter[key] = max(0, phone_counter.get(key, 0) - 1)
            if phone_counter[key] == PHONE_FRAMES:
                if frame_id - last_event_frame.get(key, -1000) > COOLDOWN_FRAMES:
                    cheat_frequency[p["seat_id"]] += 1
                    events.append([frame_id, "PHONE", p["seat_id"], "-"])
                    last_event_frame[key] = frame_id

    # ── Peeking / copying detection ──────────────────────────────
    for a in persons:
        for b in persons:
            if a["seat_id"] == b["seat_id"]:
                continue
            if abs(a["center"][1] - b["center"][1]) > ROW_THRESHOLD:
                continue
            if dist(a["center"], b["center"]) > DIST_THRESHOLD:
                continue

            paper    = paper_point(b["box"])
            to_paper = (paper[0]-a["nose"][0], paper[1]-a["nose"][1])
            ang      = angle(a["head_vec"], to_paper)
            key      = f"{a['seat_id']}-{b['seat_id']}"

            if 5 < ang < ANGLE_THRESHOLD:
                copy_counter[key] = copy_counter.get(key, 0) + 1
            else:
                copy_counter[key] = max(0, copy_counter.get(key, 0) - 1)

            if copy_counter[key] == COPY_FRAMES:
                if frame_id - last_event_frame.get(key, -1000) > COOLDOWN_FRAMES:
                    cheat_frequency[a["seat_id"]] += 1
                    events.append([frame_id, "PEEKING", a["seat_id"], b["seat_id"]])
                    last_event_frame[key] = frame_id

    # ── Annotate frame and write to video ────────────────────────
    display_frame = frame.copy()
    for p in persons:
        sid         = p["seat_id"]
        x1, y1, x2, y2 = p["box"]
        flagged     = sid in cheat_frequency and cheat_frequency[sid] > 0
        color       = (0, 0, 255) if flagged else (0, 255, 0)
        label       = f"Student {sid} - Suspect" if flagged else f"Student {sid}"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(display_frame)
    progress.progress(min(frame_id / total_frames, 1.0))

cap.release()
out.release()

# ── Persist everything in session state ───────────────────────────────────
st.session_state.cheat_frequency = dict(cheat_frequency)
st.session_state.events          = events
st.session_state.out_path        = out_path
st.session_state.video_fps       = fps
st.session_state.processing_done = True

status.success("✅ Processing completed")
progress.empty()

# Trigger a clean rerun → hits render_results() at the very top
st.rerun()