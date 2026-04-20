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
st.title("Automated Malpractice Detection System")

POSE_MODEL = "yolov8n-pose.pt"
OBJ_MODEL  = "yolov8s.pt"

FRAME_SKIP         = 5
PHONE_DETECT_GAP   = 5
CALIBRATION_FRAMES = 90
COPY_FRAMES        = 18   
PHONE_FRAMES       = 4


ANGLE_FAR  = 18   
ANGLE_NEAR = 42  
BOX_H_FAR  = 120 
BOX_H_NEAR = 280  

COOLDOWN_FRAMES    = 60
SLOW_FACTOR        = 0.4

PROCESS_W = None
PROCESS_H = None
OUTPUT_W  = None
OUTPUT_H  = None

SEAT_MERGE_RADIUS = 80


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def box_overlap_ratio(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = max((ax2-ax1)*(ay2-ay1), 1)
    areaB = max((bx2-bx1)*(by2-by1), 1)
    return inter / min(areaA, areaB)


def same_row(boxA, boxB):
    _, ay1, _, ay2 = boxA
    _, by1, _, by2 = boxB
    a_cy = (ay1 + ay2) / 2
    b_cy = (by1 + by2) / 2
    a_h  = ay2 - ay1
    b_h  = by2 - by1
   
    margin = max(a_h, b_h) * 0.40
    return abs(a_cy - b_cy) < margin


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
    best_sid  = None
    best_dist = float("inf")
    for sid, pos in st.session_state.seats.items():
        d = dist(center, pos)
        if d < best_dist:
            best_dist = d
            best_sid  = sid

    if best_dist < st.session_state.seat_merge_radius:
        old = st.session_state.seats[best_sid]
        st.session_state.seats[best_sid] = (
            int(old[0] * 0.85 + center[0] * 0.15),
            int(old[1] * 0.85 + center[1] * 0.15),
        )
        return best_sid

    new_id = st.session_state.next_seat_id
    st.session_state.seats[new_id] = center
    st.session_state.next_seat_id += 1
    return new_id


def seat_cleanup(active_ids, frame_id):
    seen = st.session_state.seat_last_seen
    for sid in list(st.session_state.seats.keys()):
        if sid not in active_ids:
            last = seen.get(sid, 0)
            if frame_id - last > 300:
                del st.session_state.seats[sid]
                seen.pop(sid, None)


def is_front_row(box, frame_height):
   
    _, y1, _, _ = box
    return y1 > frame_height * 0.45


def angle_for_height(box_h):
    
    box_h_far  = st.session_state.get("box_h_far",  BOX_H_FAR)
    box_h_near = st.session_state.get("box_h_near", BOX_H_NEAR)
    if box_h <= box_h_far:
        return ANGLE_FAR
    if box_h >= box_h_near:
        return ANGLE_NEAR
    t = (box_h - box_h_far) / (box_h_near - box_h_far)   # 0..1
    return ANGLE_FAR + t * (ANGLE_NEAR - ANGLE_FAR)


def render_results():
    cheat_frequency = st.session_state.cheat_frequency
    events          = st.session_state.events
    seats           = st.session_state.seats
    out_path        = st.session_state.out_path
    fps             = st.session_state.video_fps
    total_students  = st.session_state.total_students

    st.header("🎬 Annotated Video")
    if os.path.exists(out_path):
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(
                label="📥 Download Output Video",
                data=f,
                file_name="cheating_detection_output.mp4",
                mime="video/mp4",
                key="video_dl_btn",
            )
    else:
        st.error("❌ Output video not found")

    st.markdown(f"### 👥 Total unique students detected: **{total_students}**")

    st.header("📊 Cheating Frequency per Student")

    all_student_ids = sorted(seats.keys())
    full_freq = {sid: cheat_frequency.get(sid, 0) for sid in all_student_ids}

    freq_df = (
        pd.DataFrame(full_freq.items(), columns=["Student ID", "Cheating Count"])
        .sort_values("Student ID")
    )
    freq_df["Student Label"] = freq_df["Student ID"].apply(lambda x: f"Student {x}")

    fig1, ax1 = plt.subplots(figsize=(9, max(3, len(freq_df) * 0.55)))
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in freq_df["Cheating Count"]]
    bars = ax1.barh(freq_df["Student Label"], freq_df["Cheating Count"], color=colors)

    ax1.set_xlabel("Cheating Count")
    ax1.set_title("Cheating Frequency per Student")

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))

    for bar, val in zip(bars, freq_df["Cheating Count"]):
        ax1.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9,
        )

    ax1.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    if events:
        df = pd.DataFrame(
            events,
            columns=["Frame", "Type", "Student", "Target", "SnapshotPath"]
        )
        df["Student"] = df["Student"].apply(
            lambda x: f"Student {x}" if str(x).isdigit() else x
        )
        df["Target"] = df["Target"].apply(
            lambda x: f"Student {x}" if str(x).isdigit() else x
        )
        df["SnapshotFile"] = df["SnapshotPath"].apply(
            lambda p: os.path.basename(p) if p else ""
        )
        df_csv = df[["Frame", "Type", "Student", "Target", "SnapshotFile"]]

        st.download_button(
            "📥 Download Cheating Report (CSV)",
            data=df_csv.to_csv(index=False),
            file_name="exam_cheating_report.csv",
            mime="text/csv",
            key="csv_dl_btn",
        )

        
        snapshots_dir = st.session_state.get("snapshots_dir", "")
        if snapshots_dir and os.path.isdir(snapshots_dir):
            import zipfile, io
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                
                zf.writestr("exam_cheating_report.csv", df_csv.to_csv(index=False))
                
                for fname in sorted(os.listdir(snapshots_dir)):
                    fpath = os.path.join(snapshots_dir, fname)
                    zf.write(fpath, fname)
            zip_buf.seek(0)
            st.download_button(
                "📦 Download CSV + All Snapshot Images (ZIP)",
                data=zip_buf,
                file_name="cheating_report_with_images.zip",
                mime="application/zip",
                key="zip_dl_btn",
            )
    else:
        st.success("✅ No cheating detected")



uploaded_video = st.file_uploader(
    "Upload Exam Hall Video", type=["mp4", "avi", "mov"]
)
process_btn = st.button("▶️ Process Video")

if st.session_state.get("processing_done"):
    render_results()

if not (uploaded_video and process_btn):
    st.stop()

st.session_state.seats             = {}
st.session_state.next_seat_id      = 1
st.session_state.seat_merge_radius = SEAT_MERGE_RADIUS
st.session_state.seat_last_seen    = {}
st.session_state.processing_done   = False

status = st.empty()
status.info("🔍 Processing video… Please wait")

tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_video.read())
tfile.flush()

cap          = cv2.VideoCapture(tfile.name)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 25

PROCESS_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
PROCESS_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
OUTPUT_W  = PROCESS_W
OUTPUT_H  = PROCESS_H

status.info(
    f"🔍 Processing video… Please wait  "
    f"({PROCESS_W}×{PROCESS_H}, {total_frames} frames @ {fps:.1f} fps)"
)

fourcc   = cv2.VideoWriter_fourcc(*'avc1')
out_path      = tempfile.mktemp(suffix="_output.mp4")
snapshots_dir = tempfile.mkdtemp(prefix="cheat_snapshots_")
out      = cv2.VideoWriter(
    out_path, fourcc, max(1, fps * SLOW_FACTOR), (OUTPUT_W, OUTPUT_H)
)

pose_model = YOLO(POSE_MODEL)
obj_model  = YOLO(OBJ_MODEL)

frame_id      = 0
width_samples = []

DIST_THRESHOLD = 200

copy_counter     = {}
phone_counter    = {}
last_event_frame = {}

cheat_frequency = defaultdict(int)
events          = []
phones_last     = []


event_snapshots = {}

progress = st.progress(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    pose_res = pose_model.predict(frame, conf=0.45, verbose=False, imgsz=480)[0]
    persons  = []
    active_ids_this_frame = set()

    if pose_res.keypoints is not None:
        for i in range(len(pose_res.keypoints.xy)):
            box  = pose_res.boxes.xyxy[i]
            kpts = pose_res.keypoints.xy[i]

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            nose  = kpts[0]
            le, re       = kpts[1], kpts[2] 
            lear, rear   = kpts[3], kpts[4]   
            lsho, rsho   = kpts[5], kpts[6]

            frontrow = is_front_row((x1, y1, x2, y2), PROCESS_H)

            if frontrow:
                l_vis = lear[0] != 0
                r_vis = rear[0]  != 0

                if l_vis and r_vis:
                    ear_dx = float(rear[0] - lear[0])
                    box_w  = max(x2 - x1, 1)
                    ear_ratio = ear_dx / box_w  

                    key_cal = f"ear_cal_{i}"
                    if frame_id < CALIBRATION_FRAMES:
                        cal = st.session_state.get("ear_cal", {})
                        vals = cal.get(key_cal, [])
                        vals.append(ear_ratio)
                        cal[key_cal] = vals
                        st.session_state["ear_cal"] = cal
                        neutral_ratio = ear_ratio  
                    else:
                        cal = st.session_state.get("ear_cal", {})
                        vals = cal.get(key_cal, [ear_ratio])
                        neutral_ratio = sum(vals) / len(vals)

                    delta = ear_ratio - neutral_ratio
                    
                    hx = max(-1.0, min(1.0, -delta * 4))   
                    hy = -0.15  
                    head_vec = (hx, hy)

                elif l_vis and not r_vis:
                   
                    head_vec = (1.0, -0.1)
                elif r_vis and not l_vis:
                    
                    head_vec = (-1.0, -0.1)
                else:
                    
                    if nose[0] != 0:
                        face_pts = [k for k in [le, re] if k[0] != 0]
                        if face_pts:
                            face_mid = (
                                sum(k[0] for k in face_pts) / len(face_pts),
                                sum(k[1] for k in face_pts) / len(face_pts),
                            )
                            head_vec = (nose[0]-face_mid[0], nose[1]-face_mid[1])
                        else:
                            head_vec = (0.0, -1.0)   
                    else:
                        head_vec = (0.0, -1.0)

                if lsho[0] != 0 and rsho[0] != 0:
                    sho_dx = float(rsho[0] - lsho[0])
                    sho_dy = float(rsho[1] - lsho[1])
                    
                    sho_facing = (-sho_dy, sho_dx)
                    mag = math.hypot(*sho_facing) + 1e-6
                    sho_facing = (sho_facing[0]/mag, sho_facing[1]/mag)
                 
                    head_vec = (
                        0.6 * head_vec[0] + 0.4 * sho_facing[0],
                        0.6 * head_vec[1] + 0.4 * sho_facing[1],
                    )

                if nose[0] == 0:
                    if l_vis and r_vis:
                        nose = ((lear[0]+rear[0])/2, (lear[1]+rear[1])/2 - 10)
                    elif l_vis:
                        nose = (lear[0], lear[1] - 10)
                    elif r_vis:
                        nose = (rear[0], rear[1] - 10)
                    else:
                        nose = (cx, y1 + (y2-y1)*0.2)

            else:
            
                if nose[0] == 0:
                    if lear[0] != 0 and rear[0] != 0:
                        nose = ((lear[0]+rear[0])/2, (lear[1]+rear[1])/2 - 10)
                        head_vec = (0, -10)
                    else:
                        continue
                else:
                    face_pts = [k for k in [le, re, lear, rear] if k[0] != 0]
                    if face_pts:
                        face_mid = (
                            sum(k[0] for k in face_pts) / len(face_pts),
                            sum(k[1] for k in face_pts) / len(face_pts),
                        )
                        head_vec = (nose[0]-face_mid[0], nose[1]-face_mid[1])
                    else:
                        head_vec = (0, 10)

            seat_id = assign_seat((cx, cy))
            st.session_state.seat_last_seen[seat_id] = frame_id
            active_ids_this_frame.add(seat_id)

            box_h        = y2 - y1
            angle_thresh = angle_for_height(box_h)

            persons.append({
                "seat_id":      seat_id,
                "box":          (x1, y1, x2, y2),
                "center":       (cx, cy),
                "nose":         nose,
                "head_vec":     head_vec,
                "angle_thresh": angle_thresh,
                "box_h":        box_h,
                "is_front_row": frontrow,
                "is_active_cheat": False,
            })

            if frame_id < CALIBRATION_FRAMES:
                width_samples.append(x2 - x1)

    if frame_id % (FRAME_SKIP * 150) == 0:
        seat_cleanup(active_ids_this_frame, frame_id)

    if frame_id == CALIBRATION_FRAMES and width_samples:
        avg            = sum(width_samples) / len(width_samples)
        DIST_THRESHOLD = int(avg * 2.5)
        st.session_state.seat_merge_radius = int(avg * 0.7)

 
        avg_h = avg * 1.5
        st.session_state["box_h_far"]  = int(avg_h * 0.55)   
        st.session_state["box_h_near"] = int(avg_h * 1.55)   

    if frame_id % PHONE_DETECT_GAP == 0:
        obj_res = obj_model.predict(frame, conf=0.1, verbose=False, imgsz=640)[0]
        electronics = [] 
        for box, cls in zip(obj_res.boxes.xyxy, obj_res.boxes.cls):
            class_id = int(cls)
       
            if class_id == 67 or class_id == 68:
                px1, py1, px2, py2 = map(int, box)
                electronics.append(((px1+px2)//2, (py1+py2)//2))
        electronics_last = electronics
    else:
        electronics = phones_last 

    for p in persons:
        electronic_near_student = False
        px1, py1, px2, py2 = p["box"]
        
        padding = 40 
        for el in electronics:
            if px1 - padding < el[0] < px2 + padding and py1 - padding < el[1] < py2 + padding:
                electronic_near_student = True
                break

        key = f"elec-{p['seat_id']}"
        if electronic_near_student:
            p["is_active_cheat"] = True
            phone_counter[key] = phone_counter.get(key, 0) + 1
        else:
            phone_counter[key] = max(0, phone_counter.get(key, 0) - 1)

        if phone_counter[key] >= PHONE_FRAMES:
            if frame_id - last_event_frame.get(key, -1000) > COOLDOWN_FRAMES:
                cheat_frequency[p["seat_id"]] += 1
                events.append([frame_id, "ELECTRONICS", p["seat_id"], "-", None])
                event_snapshots[len(events) - 1] = frame.copy()
                last_event_frame[key] = frame_id

    for a in persons:
        for b in persons:
            if a["seat_id"] == b["seat_id"]:
                continue
            if dist(a["center"], b["center"]) > DIST_THRESHOLD:
                continue

            key = f"{a['seat_id']}-{b['seat_id']}"

            if box_overlap_ratio(a["box"], b["box"]) > 0.30:
                copy_counter[key] = max(0, copy_counter.get(key, 0) - 1)
                continue

            if not same_row(a["box"], b["box"]):
                copy_counter[key] = max(0, copy_counter.get(key, 0) - 1)
                continue

            paper     = paper_point(b["box"])
            to_paper  = (paper[0]-a["nose"][0], paper[1]-a["nose"][1])
            ang_paper = angle(a["head_vec"], to_paper)

            to_face  = (b["nose"][0]-a["nose"][0], b["nose"][1]-a["nose"][1])
            ang_face = angle(a["head_vec"], to_face)

            at = a["angle_thresh"]
            looking_at_paper = (5 < ang_paper < at)
            looking_at_face  = (5 < ang_face  < at)

          
            if not a["is_front_row"]:
                is_looking_down = a["head_vec"][1] > abs(a["head_vec"][0]) * 0.7 
                if is_looking_down:
                    looking_at_paper = False
                    looking_at_face  = False

            if looking_at_paper and abs(to_paper[0]) < 80 and to_paper[1] > 0:
                looking_at_paper = False

            if not a["is_front_row"]:
                head_horizontal = abs(a["head_vec"][0])
                head_vertical   = abs(a["head_vec"][1])
                if head_horizontal < head_vertical * 0.3:
                    looking_at_paper = False
                    looking_at_face  = False

            horiz_gap = abs(b["center"][0] - a["center"][0])
            vert_gap  = abs(b["center"][1] - a["center"][1]) + 1
            if horiz_gap < vert_gap * 0.25:
                looking_at_paper = False
                looking_at_face  = False

            
            if a["is_front_row"]:
                target_dx = b["center"][0] - a["center"][0]
                head_hx   = a["head_vec"][0]
                if target_dx * head_hx < 0:
                    looking_at_paper = False
                    looking_at_face  = False
                _bfar  = st.session_state.get("box_h_far",  BOX_H_FAR)
                _bnear = st.session_state.get("box_h_near", BOX_H_NEAR)
                t_h = max(0.0, min(1.0, (a["box_h"] - _bfar) / max(_bnear - _bfar, 1)))
                min_hx = 0.20 - t_h * 0.10   
                if abs(head_hx) < min_hx:
                    looking_at_paper = False
                    looking_at_face  = False

            if looking_at_paper or looking_at_face:
                a["is_active_cheat"] = True
                copy_counter[key] = copy_counter.get(key, 0) + 1
            else:
                copy_counter[key] = max(0, copy_counter.get(key, 0) - 1)

            if copy_counter[key] >= COPY_FRAMES:
                if frame_id - last_event_frame.get(key, -1000) > COOLDOWN_FRAMES:
                    cheat_frequency[a["seat_id"]] += 1
                    events.append([frame_id, "PEEKING", a["seat_id"], b["seat_id"], None])
                    event_snapshots[len(events) - 1] = frame.copy()
                    last_event_frame[key] = frame_id

    display_frame = frame.copy()
    for p in persons:
        sid            = p["seat_id"]
        x1, y1, x2, y2 = p["box"]

        is_flagged = (cheat_frequency.get(sid, 0) > 0) or p["is_active_cheat"]
        color = (0, 0, 255) if is_flagged else (0, 255, 0)
        label = f"S{sid}"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total_so_far = len(st.session_state.seats)
    cv2.putText(
        display_frame,
        f"Students: {len(persons)} visible | {total_so_far} total",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
    )

    out.write(display_frame)
    progress.progress(min(frame_id / total_frames, 1.0))

cap.release()
out.release()

for idx, snap_frame in event_snapshots.items():
    ann = snap_frame.copy()
    f_id, e_type, s_id, tgt, _ = events[idx]

    cv2.rectangle(ann, (0, 0), (ann.shape[1], 40), (0, 0, 200), -1)
    cv2.putText(ann, f"{e_type}  Student {s_id}  Frame {f_id}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    img_name = f"event_{idx:04d}_{e_type}_S{s_id}_frame{f_id}.jpg"
    img_path = os.path.join(snapshots_dir, img_name)
    cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
    events[idx][4] = img_path

MIN_FRAMES_TO_BE_A_STUDENT = 5
valid_seats = {
    sid: pos
    for sid, pos in st.session_state.seats.items()
    if st.session_state.seat_last_seen.get(sid, 0) > MIN_FRAMES_TO_BE_A_STUDENT
}
id_remap    = {old: new for new, old in enumerate(sorted(valid_seats.keys()), 1)}
final_seats = {id_remap[sid]: pos for sid, pos in valid_seats.items()}

final_cheat = {id_remap[sid]: cnt
               for sid, cnt in cheat_frequency.items()
               if sid in id_remap}
final_events = [
    [
        f, t,
        id_remap.get(int(s), s),
        id_remap.get(int(tgt), tgt) if str(tgt).isdigit() else tgt,
        img_path,
    ]
    for f, t, s, tgt, img_path in events
    if str(s).isdigit() and int(s) in id_remap
]

st.session_state.cheat_frequency = final_cheat
st.session_state.events          = final_events
st.session_state.seats           = final_seats
st.session_state.out_path        = out_path
st.session_state.video_fps       = fps
st.session_state.total_students  = len(final_seats)
st.session_state.snapshots_dir   = snapshots_dir
st.session_state.processing_done = True

status.success(f"✅ Processing completed — {len(final_seats)} students detected")
progress.empty()

st.rerun()