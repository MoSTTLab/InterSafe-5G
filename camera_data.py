"""
camera_data.py
---------------
What this module does:
    1. Captures RTSP frames
    2. Runs YOLO detection + tracking
    3. Computes world position (homography), lat/lon, speed, bearing
    4. Runs pose estimation for pedestrians
    5. Packs everything into a TrackObject dict and pushes it to `track_queue`

─────────────────────────────────────────────────────────────
Queue item format  (one dict per tracked object per frame)
─────────────────────────────────────────────────────────────
{
    "frame_no":   int,           # frame counter (skipped frames excluded)
    "timestamp":  float,         # frame_no / fps (seconds from stream start)
    "obj_id":     int,           # YOLO tracker ID
    "label":      str,           # canonical class name from CLASS_MAPPING
    "conf":       float,         # detection confidence
    "bbox":       (x1,y1,x2,y2),# pixel bounding box
    "foot_px":    (px, py),      # bottom-centre pixel (used for homography)
    "X_world":    float,         # smoothed world X (metres)
    "Y_world":    float,         # smoothed world Y (metres)
    "dx":         float,         # world displacement vector X since oldest history
    "dy":         float,         # world displacement vector Y since oldest history
    "speed_m_s":  float,         # instantaneous speed in m/s
    "speed_km_h": float,         # same in km/h
    "bearing":    float | None,  # compass bearing in degrees (None if stationary)
    "direction":  str,           # human-readable reduced bearing
    "latitude":   float,         # GPS latitude
    "longitude":  float,         # GPS longitude
    # pose fields — only populated for Pedestrian; empty strings otherwise
    "activity":   str,           # "walking" | "running" | "standing" | "unknown" | ""
    "body_lean":  str,           # "left" | "right" | "straight" | ""
    "move_dir":   str,           # "left" | "right" | "straight" | ""
    "kp_xy":      np.ndarray | None,   # 17×2 keypoint coordinates (full frame)
    "kp_conf":    np.ndarray | None,   # 17 keypoint confidences
}

─────────────────────────────────────────────────────────────
Also exported (for processing code (main_script.py) to use):
─────────────────────────────────────────────────────────────
    track_queue      — queue.Queue of the dicts above
    conflict_points  — list of (X, Y) world-coord conflict points
    fps              — stream FPS (set after start_pipeline() returns)
    get_heading()    — (py_history, min_py_delta) -> "toward"/"away"/"unknown";
                       standalone, importable, callable from main_script too
    start_pipeline() — starts capture+detection in background thread
    stop_pipeline()  — signals clean shutdown
"""

import csv
import math
import os
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
import cv2
import numpy as np
import torch
import json
from ultralytics import YOLO

# =============================================================================
# CONFIGURATIONS - PATHS 
# =============================================================================
MODEL_PATH      = ""
POSE_MODEL_PATH = ""
HOMOGRAPHY_JSON = ""
RTSP_URL        = "" 

# =============================================================================
# CONFIGURATIONS - CONSTANTS
# =============================================================================
REFERENCE_LAT   = 13.051754
REFERENCE_LON   = 77.540361
ORIENTATION_DEG = -90.0

TRACKER         = "botsort.yaml"
FRAME_SKIP      = 1
MIN_DT          = 0.05
ALPHA_POS       = 0.5
TRACKLET_LENGTH = 50
QUEUE_SIZE      = 20   # max objects waiting; drops oldest if full
MIN_SPEED_FOR_TTRC = 1.0

# ── Pose config ───────────────────────────────────────────────────────────────
ENABLE_POSE         = True
POSE_EVERY_N        = 1       # throttle: run pose at most every N frames per track
POSE_CROP_PAD       = 10      # px padding around pedestrian crop
MIN_POSE_KEYPOINTS  = 4       # visible KPs required to confirm pedestrian
POSE_CONF_THRESHOLD = 0.2

# ── Display flags (set False to disable window) ───────────────────────────────
SHOW_VIDEO      = True
DRAW_BOXES      = True
DRAW_ROI        = True
DRAW_CP         = True
DRAW_TRACKLETS  = False
DRAW_SKELETON   = False

# =============================================================================
# CLASS MAPPING 
# =============================================================================
CLASS_MAPPING = {
    "Van":         "Car",
    "Sedan":       "Car",
    "MUV":         "Car",
    "SUV":         "Car",
    "Hatchback":   "Car",
    "Mini-bus":    "Bus",
    "LCV":         "Heavy Vehicle",
    "Truck":       "Heavy Vehicle",
    "Mini-truck":  "Heavy Vehicle",
}

CLASS_COLORS = {
    "Pedestrian":    (0, 0, 0),
    "Bicycle":       (255, 180, 0),
    "Two-wheeler":   (0, 255, 180),
    "Car":           (255, 255, 255),
    "Bus":           (180, 0, 255),
    "Heavy Vehicle": (0, 80, 255),
    "Three-wheeler": (90, 78, 200),
    "Tempo Traveller":(9, 210, 0)
}

# =============================================================================
# POSE CONSTANTS  (COCO 17-keypoint indices)
# =============================================================================
KP_NOSE        = 0
KP_LEFT_EYE    = 1;  KP_RIGHT_EYE   = 2
KP_LEFT_EAR    = 3;  KP_RIGHT_EAR   = 4
KP_LEFT_SHLDR  = 5;  KP_RIGHT_SHLDR = 6
KP_LEFT_ELBOW  = 7;  KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST  = 9;  KP_RIGHT_WRIST = 10
KP_LEFT_HIP    = 11; KP_RIGHT_HIP   = 12
KP_LEFT_KNEE   = 13; KP_RIGHT_KNEE  = 14
KP_LEFT_ANKLE  = 15; KP_RIGHT_ANKLE = 16

SKELETON_PAIRS = [
    (KP_NOSE, KP_LEFT_EYE),      (KP_NOSE, KP_RIGHT_EYE),
    (KP_LEFT_EYE, KP_LEFT_EAR),  (KP_RIGHT_EYE, KP_RIGHT_EAR),
    (KP_LEFT_SHLDR, KP_RIGHT_SHLDR),
    (KP_LEFT_SHLDR, KP_LEFT_ELBOW),   (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHLDR, KP_RIGHT_ELBOW), (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHLDR, KP_LEFT_HIP),     (KP_RIGHT_SHLDR, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_LEFT_KNEE),      (KP_LEFT_KNEE, KP_LEFT_ANKLE),
    (KP_RIGHT_HIP, KP_RIGHT_KNEE),    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
]

# =============================================================================
# SHARED EXPORTS  (processing code imports these)
# =============================================================================
track_queue     = queue.Queue(maxsize=QUEUE_SIZE)
conflict_points = []   # populated after load_homography() inside start_pipeline()
fps             = 15.0 # updated after cap.get(CAP_PROP_FPS); camera is natively 15fps

# Per-object position history for TTRC — keyed by obj_id
#_ttrc_history: dict = defaultdict(lambda: deque(maxlen=50))

# Internal control
_stop_event = threading.Event()

# =============================================================================
# CAMERA DETECTION CSV LOG
# =============================================================================
_CAM_CSV_DIR = "/home/coewwt/Videos/intersafe_recordings"
os.makedirs(_CAM_CSV_DIR, exist_ok=True)

_cam_session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_CAM_CSV_PATH   = os.path.join(_CAM_CSV_DIR, f"camera_tracks_{_cam_session_ts}.csv")

_CAM_CSV_COLUMNS = [
    "wall_time",     # real wall-clock time — correlate with alerts_*.csv
    "frame_no",      # processed frame counter (skipped frames excluded)
    "stream_ts_s",   # frame_no / fps — seconds from stream start
    "obj_id",        # YOLO BotSORT tracker ID
    "label",         # class name (from CLASS_MAPPING)
    "conf",          # YOLO detection confidence (0–1)
    "foot_px_x",     # foot-point pixel x (bottom-centre of bbox)
    "foot_px_y",     # foot-point pixel y
    "X_world_m",     # homography world X in metres
    "Y_world_m",     # homography world Y in metres
    "speed_km_h",    # instantaneous speed
    "bearing_deg",   # compass bearing in degrees (blank if stationary)
    "direction",     # human-readable reduced bearing e.g. "NE"
    "latitude",      # GPS latitude
    "longitude",     # GPS longitude
    "ttrc_s",        # seconds to conflict point; blank if not heading toward CP
    "activity",      # pose: walking/running/standing/unknown  (pedestrians only)
    "body_lean",     # pose: left/right/straight               (pedestrians only)
    "gate_status",   # "approved" on first appearance, "tracking" after
]

_cam_csv_lock   = threading.Lock()
_cam_csv_file   = open(_CAM_CSV_PATH, "w", newline="")
_cam_csv_writer = csv.DictWriter(_cam_csv_file, fieldnames=_CAM_CSV_COLUMNS)
_cam_csv_writer.writeheader()
_cam_csv_file.flush()
print(f"[CAM-CSV] Detection log → {_CAM_CSV_PATH}")


# =============================================================================
# UTILITY FUNCTIONS  (geometry, geo-conversion, bearing)
# =============================================================================

def load_homography(path):
    with open(path, "r") as f:
        data = json.load(f)
    H   = np.array(data["H"], dtype=np.float64)
    roi = (np.array(data["roi1_image_points"], dtype=np.int32)
           if "roi1_image_points" in data else None)
    cps = [tuple(pt) for pt in data["conflict_world_points"]]
    return H, roi, cps


def pixel_to_world(px, py, H):
    p  = np.array([px, py, 1.0])
    wp = H @ p
    if abs(wp[2]) < 1e-9:
        return None
    wp /= wp[2]
    return float(wp[0]), float(wp[1])


def world_to_pixel(X, Y, H_inv):
    p  = H_inv @ np.array([X, Y, 1.0])
    p /= p[2]
    return int(p[0]), int(p[1])


def rotate_to_EN(x, y, orientation_deg):
    theta = math.radians(orientation_deg)
    c, s  = math.cos(theta), math.sin(theta)
    return c*x + s*y, -s*x + c*y


def meters_to_latlon(east, north, ref_lat, ref_lon):
    phi           = math.radians(ref_lat)
    m_per_deg_lat = 111132.954 - 559.822*math.cos(2*phi) + 1.175*math.cos(4*phi)
    m_per_deg_lon = 111412.84*math.cos(phi) - 93.5*math.cos(3*phi)
    return ref_lat + north/m_per_deg_lat, ref_lon + east/m_per_deg_lon


def world_to_latlon(X, Y):
    east, north = rotate_to_EN(X, Y, ORIENTATION_DEG)
    return meters_to_latlon(east, north, REFERENCE_LAT, REFERENCE_LON)


def compute_bearing(dx, dy):
    east, north = rotate_to_EN(dx, dy, ORIENTATION_DEG)
    if abs(east) < 1e-6 and abs(north) < 1e-6:
        return None
    return (math.degrees(math.atan2(east, north)) + 360) % 360


def reduced_bearing(b):
    if b is None:  return "stationary"
    if b < 90:     return f"N {b:.1f}° E"
    if b < 180:    return f"S {180-b:.1f}° E"
    if b < 270:    return f"S {b-180:.1f}° W"
    return         f"N {360-b:.1f}° W"


# =============================================================================
# POSE FUNCTIONS
# =============================================================================

def _get_kp(kp_xy, kp_conf, idx):
    """Safe keypoint accessor. Returns (x, y, conf) or None."""
    if kp_xy is None or kp_conf is None or idx >= len(kp_xy):
        return None
    x, y = float(kp_xy[idx][0]), float(kp_xy[idx][1])
    conf = float(kp_conf[idx])
    return (x, y, conf) if conf >= POSE_CONF_THRESHOLD else None


def count_visible_keypoints(kp_conf):
    if kp_conf is None:
        return 0
    return int((kp_conf >= POSE_CONF_THRESHOLD).sum())


def run_pose_on_pedestrian(frame, x1, y1, x2, y2, frame_w, frame_h, pose_model):
    """Crop + run pose model; remap keypoints to full-frame coords."""
    cx1  = max(x1 - POSE_CROP_PAD, 0)
    cy1  = max(y1 - POSE_CROP_PAD, 0)
    cx2  = min(x2 + POSE_CROP_PAD, frame_w - 1)
    cy2  = min(y2 + POSE_CROP_PAD, frame_h - 1)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, None

    pose_res = pose_model(crop, imgsz=256, verbose=False, conf=0.3, stream=False)
    if (not pose_res
            or pose_res[0].keypoints is None
            or len(pose_res[0].keypoints.xy) == 0):
        return None, None

    kpts     = pose_res[0].keypoints
    best_idx = (int(kpts.conf.mean(dim=1).argmax().item())
                if kpts.conf is not None else 0)

    kp_xy_crop  = kpts.xy[best_idx].cpu().numpy()
    kp_conf_out = (kpts.conf[best_idx].cpu().numpy()
                   if kpts.conf is not None
                   else np.ones(len(kp_xy_crop), dtype=np.float32))

    kp_xy_full        = kp_xy_crop.copy()
    kp_xy_full[:, 0] += cx1
    kp_xy_full[:, 1] += cy1
    return kp_xy_full, kp_conf_out


def classify_pedestrian_activity(kp_xy, kp_conf):
    """Returns (activity, body_lean)."""
    def kp(idx): return _get_kp(kp_xy, kp_conf, idx)

    l_shldr = kp(KP_LEFT_SHLDR);  r_shldr = kp(KP_RIGHT_SHLDR)
    l_hip   = kp(KP_LEFT_HIP);    r_hip   = kp(KP_RIGHT_HIP)
    l_knee  = kp(KP_LEFT_KNEE);   r_knee  = kp(KP_RIGHT_KNEE)
    l_ankle = kp(KP_LEFT_ANKLE);  r_ankle = kp(KP_RIGHT_ANKLE)

    torso_h = None
    if l_shldr and l_hip:     torso_h = abs(l_hip[1] - l_shldr[1])
    elif r_shldr and r_hip:   torso_h = abs(r_hip[1] - r_shldr[1])
    if not torso_h or torso_h < 10:
        return "unknown", "straight"

    knee_lifts = [max(hip[1] - knee[1], 0)
                  for hip, knee in [(l_hip, l_knee), (r_hip, r_knee)]
                  if hip and knee]
    mean_klr = (sum(knee_lifts)/len(knee_lifts))/torso_h if knee_lifts else 0

    ankle_spread = 0.0
    if l_ankle and r_ankle:
        hip_w        = max(abs(l_hip[0]-r_hip[0]), 1.0) if (l_hip and r_hip) else 1.0
        ankle_spread = abs(l_ankle[0]-r_ankle[0])/hip_w

    if mean_klr > 0.35:        activity = "running"
    elif ankle_spread < 0.40:  activity = "standing"
    else:                      activity = "walking"

    shldr_mid_x = ((l_shldr[0]+r_shldr[0])/2 if (l_shldr and r_shldr)
                   else (l_shldr[0] if l_shldr else (r_shldr[0] if r_shldr else None)))
    hip_mid_x   = ((l_hip[0]+r_hip[0])/2 if (l_hip and r_hip)
                   else (l_hip[0] if l_hip else (r_hip[0] if r_hip else None)))

    body_lean = "straight"
    if shldr_mid_x is not None and hip_mid_x is not None:
        torso_w      = max(abs(l_shldr[0]-r_shldr[0]), 1.0) if (l_shldr and r_shldr) else 1.0
        offset_ratio = (shldr_mid_x - hip_mid_x) / torso_w
        if offset_ratio < -0.15:  body_lean = "left"
        elif offset_ratio > 0.15: body_lean = "right"

    return activity, body_lean


def classify_pedestrian_move_direction(kp_xy, kp_conf):
    """Returns 'left' | 'right' | 'straight' from leading foot placement."""
    def kp(idx): return _get_kp(kp_xy, kp_conf, idx)

    l_ankle = kp(KP_LEFT_ANKLE);  r_ankle = kp(KP_RIGHT_ANKLE)
    l_hip   = kp(KP_LEFT_HIP);    r_hip   = kp(KP_RIGHT_HIP)
    if not (l_ankle and r_ankle): return "straight"

    hip_mid_x = ((l_hip[0]+r_hip[0])/2 if (l_hip and r_hip)
                 else (l_hip[0] if l_hip else (r_hip[0] if r_hip else None)))
    if hip_mid_x is None: return "straight"

    leading_x = l_ankle[0] if l_ankle[1] > r_ankle[1] else r_ankle[0]
    hip_w     = max(abs(l_hip[0]-r_hip[0]), 1.0) if (l_hip and r_hip) else 1.0
    offset    = (leading_x - hip_mid_x) / hip_w

    if offset < -0.2:  return "left"
    if offset > 0.2:   return "right"
    return "straight"


def draw_skeleton(frame, kp_xy, kp_conf, color=(0, 255, 128)):
    for (i, j) in SKELETON_PAIRS:
        a = _get_kp(kp_xy, kp_conf, i)
        b = _get_kp(kp_xy, kp_conf, j)
        if a and b:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 2)
    for idx in range(len(kp_xy)):
        kp = _get_kp(kp_xy, kp_conf, idx)
        if kp:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 255, 0), -1)


# =============================================================================
# TTRC COMPUTATION
# =============================================================================

def compute_ttrc(x_last, y_last, dx, dy, speed_m_s, conflict_points):
    if speed_m_s < 1e-6:
        return None
    best = None
    for (x_CP, y_CP) in conflict_points:
        dx_CP = x_CP - x_last
        dy_CP = y_CP - y_last
        if dx * dx_CP + dy * dy_CP <= 0:
            continue
        t = math.hypot(dx_CP, dy_CP) / speed_m_s
        if best is None or t < best["TTRC"]:
            best = {"TTRC": t, "cp": (x_CP, y_CP)}
    return best


# Global label lookup so _compute_ttrc can access the label by obj_id
label_for_id: dict = {}


def get_heading(py_history, min_py_delta: int = 6) -> str:
    """
    Args:
        py_history:   sequence of py pixel values, oldest → newest
                      (e.g. a list or deque of recent foot-point y's).
        min_py_delta: minimum net pixel movement required to call a
                      direction, filtering out detection jitter.

    Returns:
        "toward"  — net downward movement >= min_py_delta
        "away"    — net upward movement >= min_py_delta
        "unknown" — fewer than 2 samples, or movement is within the
                    jitter band (net |Δpy| < min_py_delta)
    """
    if len(py_history) < 2:
        return "unknown"

    py_delta = py_history[-1] - py_history[0]

    if py_delta >= min_py_delta:
        return "toward"
    if py_delta <= -min_py_delta:
        return "away"
    return "unknown"

# =============================================================================
# MODEL DEVICE PLACEMENT
# =============================================================================

def _move_model_to_device(model: "YOLO", device: str, model_name: str) -> str:
    """
    Move a YOLO model onto `device`, logging (not swallowing) any failure and
    explicitly forcing the model onto CPU afterward instead of trusting
    whatever partial state a failed .to() call may have left it in.

    Returns the device that should be used going forward — either the
    requested `device`, or "cpu" if the move failed. Callers should feed
    this return value into the next _move_model_to_device() call so a GPU
    failure on one model doesn't leave later models trying (and re-logging
    a failure) on a device that's already known to be bad.
    """
    if device == "cpu":
        model.model.to("cpu")
        return "cpu"

    try:
        model.model.to(device)
        return device
    except (RuntimeError, AssertionError, AttributeError) as exc:
        print(f"[CAM] ⚠️  {model_name} model → {device} failed, "
              f"falling back to CPU: {exc}")
        try:
            model.model.to("cpu")
        except Exception as cpu_exc:
            # Extremely unlikely, but don't let a broken .to("cpu") crash the pipeline thread on top of the original GPU failure.
            print(f"[CAM] ⚠️  {model_name} model → cpu fallback also "
                  f"failed: {cpu_exc}")
        return "cpu"


# =============================================================================
# MAIN PIPELINE LOOP  (runs in a background thread)
# =============================================================================

def _pipeline_loop():
    global fps, conflict_points

    # ── Load models ───────────────────────────────────────────────────────────
    print("[CAM] Loading detection model …")
    model  = YOLO(MODEL_PATH)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[CAM] Using device: {device}")
    device = _move_model_to_device(model, device, "detection")

    pose_model = None
    if ENABLE_POSE:
        print("[CAM] Loading pose model …")
        pose_model = YOLO(POSE_MODEL_PATH)
        device = _move_model_to_device(pose_model, device, "pose")

    id_to_label = {i: CLASS_MAPPING.get(name, name) for i, name in model.names.items()}

    # ── Load homography ───────────────────────────────────────────────────────
    print("[CAM] Loading homography …")
    H, ROI1, cps = load_homography(HOMOGRAPHY_JSON)
    H_inv        = np.linalg.inv(H)
    conflict_points[:] = cps   # update the module-level list in-place

    roi_draw     = ROI1.reshape((-1, 1, 2)) if ROI1 is not None else None
    cp_img_coords = []
    for (x_CP, y_CP) in conflict_points:
        p  = H_inv @ np.array([x_CP, y_CP, 1.0])
        p /= p[2]
        cp_img_coords.append((int(p[0]), int(p[1])))

    # ── Open stream ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Camera is natively 15fps — do not request a higher FPS here; asking an
    # RTSP source for a rate it can't deliver either does nothing or triggers
    # renegotiation stalls, and it makes cap.get(CAP_PROP_FPS) below unreliable.

    if not cap.isOpened():
        print("[CAM] ❌  Cannot open RTSP stream.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Stream opened  (FPS: {fps:.1f}, {w}×{h})")

    if SHOW_VIDEO:
        cv2.namedWindow("Camera — iSafe", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera — iSafe", 1280, 720)

    # ── Per-track state ───────────────────────────────────────────────────────
    history      = defaultdict(lambda: deque(maxlen=TRACKLET_LENGTH))
    pos_smooth   = defaultdict(lambda: None)
    approved_ids = set()

    # ── ROI gate state ─────────────────────────────────────────────────────────
    # Simple two-step rule:
    #   1. Is the object's foot point inside the ROI right now?
    #        No  → not tracked at all, nothing recorded.
    #   2. If yes, is it heading toward the camera or away from it?
    #        Away  → not detected/tracked.
    #        Toward → approved: tracked and drawn from here on, and stays
    #                 approved even after it later leaves the ROI.
    #
    # "Heading" is judged from the foot point's vertical pixel movement (py)
    # while inside the ROI and not yet approved: increasing py = getting
    # lower in frame = moving toward the camera; decreasing py = moving away.
    # A small window + threshold avoids single-frame detection jitter being
    # mistaken for real movement.
    ROI_HEADING_WINDOW = 5   # frames of py history to judge heading, while inside ROI
    ROI_MIN_PY_DELTA    = 6  # pixels, net vertical movement needed to call it a direction

    # roi_heading_hist: obj_id → deque of py samples, collected only while the
    # object is inside the ROI and not yet approved. Cleared once approved or
    # once the object leaves the ROI before a direction was ever confirmed.
    roi_heading_hist: dict[int, deque] = {}

    # Pose state
    pose_cache         = {}               # obj_id → {activity, body_lean, move_dir, kp_xy, kp_conf}
    pose_frame_counter = defaultdict(int)
    pose_confirmed_ids = set()

    # Display-only (alert overlay from processing code via alert_status — optional)
    alert_status = {}   # obj_id → "Reaching CP" / "Reached CP"  (set by processing code if needed)

    frame_no = 0

    # ── Main capture + detection loop ─────────────────────────────────────────
    while not _stop_event.is_set():

        cap.grab()
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            print("[CAM] ⚠️  Empty frame — reconnecting in 2 s …")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            new_fps = cap.get(cv2.CAP_PROP_FPS)
            if new_fps > 0:
                fps = new_fps
            frame_no = 0

            # On reconnect the tracker will reassign IDs from scratch.
            # Clear all gate state so revived IDs get a fair heading check.
            approved_ids.clear()
            roi_heading_hist.clear()
            history.clear()
            pos_smooth.clear()
            pose_cache.clear()
            pose_frame_counter.clear()
            pose_confirmed_ids.clear()
            print("[CAM] Stream reconnected — gate and track state reset.")
            continue

        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        ts = frame_no / fps

        # ── YOLO detect + track ───────────────────────────────────────────────
        results = model.track(frame, tracker=TRACKER,
                              persist=True, imgsz=960, verbose=False, conf = 0.2)

        if results and results[0].boxes is not None:

            _newly_approved = set()   # obj_ids approved for the first time this frame

            for box in results[0].boxes:

                if box.id is None:
                    continue

                obj_id = int(box.id.item())
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = id_to_label.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                px, py          = int((x1+x2)/2), y2   # foot point

                # ── ROI gate  (inside ROI, then heading check) ────────────────
                #
                # Step 1: is the foot point inside the ROI right now?
                # Step 2: if yes and not yet approved, is it heading toward
                #         the camera (py increasing) or away (py decreasing)?
                # Once approved, an object stays approved and keeps being
                # tracked even after it later leaves the ROI.

                inside_roi = (ROI1 is not None and
                              cv2.pointPolygonTest(ROI1, (float(px), float(py)), False) >= 0)

                if obj_id not in approved_ids:

                    if not inside_roi:
                        # Not inside the ROI — nothing to evaluate or track yet.
                        roi_heading_hist.pop(obj_id, None)
                        continue

                    # Inside the ROI — build a short py history to judge heading.
                    hist_py = roi_heading_hist.setdefault(
                        obj_id, deque(maxlen=ROI_HEADING_WINDOW))
                    hist_py.append(py)

                    heading = get_heading(hist_py, ROI_MIN_PY_DELTA)

                    if heading == "toward":
                        approved_ids.add(obj_id)
                        _newly_approved.add(obj_id)
                        roi_heading_hist.pop(obj_id, None)
                        print(f"[CAM] ID:{obj_id} approved — inside ROI, "
                              f"heading toward camera")
                    else:
                        # "away"    → clearly leaving the camera, not a hazard.
                        # "unknown" → not enough samples / within jitter band.
                        # Either way: not yet approved, keep/skip for now.
                        continue

                # ── Homography → world coords ─────────────────────────────────
                world = pixel_to_world(px, py, H)
                if world is None:
                    continue
                X_raw, Y_raw = world

                # ── EMA position smoothing ────────────────────────────────────
                if pos_smooth[obj_id] is None:
                    X, Y = X_raw, Y_raw
                else:
                    X_prev, Y_prev = pos_smooth[obj_id]
                    X = ALPHA_POS * X_raw + (1 - ALPHA_POS) * X_prev
                    Y = ALPHA_POS * Y_raw + (1 - ALPHA_POS) * Y_prev

                pos_smooth[obj_id] = (X, Y)
                history[obj_id].append((frame_no, ts, X, Y))

                # ── Speed + bearing ───────────────────────────────────────────
                speed_m_s = 0.0
                bearing   = None
                dx = dy   = 0.0

                hist = history[obj_id]
                if len(hist) >= 2:
                    f_oldest, _, x_oldest, y_oldest = hist[0]
                    dt  = max((frame_no - f_oldest) / fps, MIN_DT)
                    dx  = X - x_oldest
                    dy  = Y - y_oldest
                    speed_m_s = math.hypot(dx, dy) / dt
                    bearing   = compute_bearing(dx, dy)

                speed_km_h = speed_m_s * 3.6
                direction  = reduced_bearing(bearing)
                lat, lon   = world_to_latlon(X, Y)

                # ── TTRC computation ──────────────────────────────────────────
                ttrc = None
                if len(hist) >= 2 and speed_m_s >= MIN_SPEED_FOR_TTRC:
                    result = compute_ttrc(X, Y, dx, dy, speed_m_s, conflict_points)
                    if result is not None:
                        ttrc = round(result["TTRC"], 3)
                
                # ── Pose estimation (Pedestrian only) ─────────────────────────
                activity  = ""
                body_lean = ""
                move_dir  = ""
                kp_xy     = None
                kp_conf   = None

                if label == "Pedestrian" and ENABLE_POSE and pose_model is not None:
                    counter      = pose_frame_counter[obj_id]
                    run_pose_now = (counter % POSE_EVERY_N == 0)
                    pose_frame_counter[obj_id] = counter + 1

                    if run_pose_now:
                        kp_xy_new, kp_conf_new = run_pose_on_pedestrian(
                            frame, x1, y1, x2, y2, w, h, pose_model)
                        if kp_xy_new is not None:
                            act, lean = classify_pedestrian_activity(kp_xy_new, kp_conf_new)
                            mdir      = classify_pedestrian_move_direction(kp_xy_new, kp_conf_new)
                            pose_cache[obj_id] = {
                                "activity":  act,
                                "body_lean": lean,
                                "move_dir":  mdir,
                                "kp_xy":     kp_xy_new,
                                "kp_conf":   kp_conf_new,
                            }
                            if count_visible_keypoints(kp_conf_new) >= MIN_POSE_KEYPOINTS:
                                pose_confirmed_ids.add(obj_id)

                    cached    = pose_cache.get(obj_id, {})
                    activity  = cached.get("activity",  "")
                    body_lean = cached.get("body_lean", "")
                    move_dir  = cached.get("move_dir",  "")
                    kp_xy     = cached.get("kp_xy",     None)
                    kp_conf   = cached.get("kp_conf",   None)

                # ── Build TrackObject dict ────────────────────────────────────
                track_obj = {
                    "frame_no":   frame_no,
                    "timestamp":  ts,
                    "obj_id":     obj_id,
                    "label":      label,
                    "conf":       conf,
                    "bbox":       (x1, y1, x2, y2),
                    "foot_px":    (px, py),
                    "X_world":    X,
                    "Y_world":    Y,
                    "dx":         dx,
                    "dy":         dy,
                    "speed_m_s":  speed_m_s,
                    "speed_km_h": speed_km_h,
                    "bearing":    bearing,
                    "direction":  direction,
                    "latitude":   lat,
                    "longitude":  lon,
                    # pose (empty strings / None for non-pedestrians)
                    "activity":   activity,
                    "body_lean":  body_lean,
                    "move_dir":   move_dir,
                    "kp_xy":      kp_xy,
                    "kp_conf":    kp_conf,
                    # TTRC — None if object not heading toward any conflict point
                    "ttrc":       ttrc,
                }

                # Drop oldest if queue is full (stay live)
                if track_queue.full():
                    try:
                        track_queue.get_nowait()
                    except queue.Empty:
                        pass

                track_queue.put(track_obj)

                # ── Camera detection CSV log ──────────────────────────────────
                with _cam_csv_lock:
                    _cam_csv_writer.writerow({
                        "wall_time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "frame_no":     frame_no,
                        "stream_ts_s":  round(ts, 3),
                        "obj_id":       obj_id,
                        "label":        label,
                        "conf":         round(conf, 3),
                        "foot_px_x":    px,
                        "foot_px_y":    py,
                        "X_world_m":    round(X, 3),
                        "Y_world_m":    round(Y, 3),
                        "speed_km_h":   round(speed_km_h, 2),
                        "bearing_deg":  round(bearing, 1) if bearing is not None else "",
                        "direction":    direction,
                        "latitude":     round(lat, 7),
                        "longitude":    round(lon, 7),
                        "ttrc_s":       round(ttrc, 3) if ttrc is not None else "",
                        "activity":     activity,
                        "body_lean":    body_lean,
                        "gate_status":  "approved" if obj_id in _newly_approved else "tracking",
                    })
                    _cam_csv_file.flush()

                # ── Optional display ──────────────────────────────────────────
                if SHOW_VIDEO and DRAW_BOXES:
                    color       = CLASS_COLORS.get(label, (200, 200, 200))
                    status_text = alert_status.get(obj_id, "")
                    if label == "Pedestrian":
                        lbl = f"Ped ID:{obj_id} {direction} {speed_km_h:.1f}km/h {activity} {body_lean}"
                    else:
                        lbl = f"{label} ID:{obj_id} {direction} {speed_km_h:.1f}km/h"
                    if status_text:
                        lbl += f" | {status_text}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, lbl, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    if DRAW_SKELETON and label == "Pedestrian" and kp_xy is not None:
                        draw_skeleton(frame, kp_xy, kp_conf)

        # ── Static geometry overlay ───────────────────────────────────────────
        if SHOW_VIDEO:
            if DRAW_TRACKLETS:
                for tid, hist in history.items():
                    if len(hist) < 2:
                        continue
                    pts   = [world_to_pixel(X, Y, H_inv) for (_, _, X, Y) in hist]
                    color = (CLASS_COLORS.get(
                        "Pedestrian", (0, 255, 0)))  # simple fallback
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i-1], pts[i], color,
                                 int(1 + 2*i/len(pts)))

            if DRAW_ROI and roi_draw is not None:
                cv2.polylines(frame, [roi_draw], True, (0, 255, 255), 2)

            if DRAW_CP:
                for (cx, cy) in cp_img_coords:
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "CP", (cx+5, cy-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Camera — iSafe", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _stop_event.set()
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    with _cam_csv_lock:
        _cam_csv_file.flush()
        _cam_csv_file.close()
    print(f"[CAM-CSV] Detection log closed → {_CAM_CSV_PATH}")
    print("[CAM] Pipeline thread exited.")


# =============================================================================
# PUBLIC API
# =============================================================================

def start_pipeline():
    """
    Start the camera + detection pipeline in a background thread.
    Returns the thread object.
    Call stop_pipeline() to shut it down cleanly.
    """
    _stop_event.clear()
    t = threading.Thread(target=_pipeline_loop, daemon=True, name="CamPipeline")
    t.start()
    print("[CAM] Pipeline started.")
    return t


def stop_pipeline():
    """Signal the pipeline thread to stop."""
    _stop_event.set()
    print("[CAM] Stop signal sent.")


# =============================================================================
# STANDALONE TEST  —  just verifies the pipeline pushes data to the queue
# =============================================================================
if __name__ == "__main__":
    t = start_pipeline()
    print("[TEST] Waiting for track objects … (Ctrl+C to quit)")
    try:
        while True:
            try:
                obj = track_queue.get(timeout=3)
                print(
                    f"[TRACK] Frame#{obj['frame_no']:05d}  "
                    f"Timestamp = {obj['timestamp']}"
                    f"ID:{obj['obj_id']:3d}  {obj['label']:12s}  "
                    f"{obj['speed_km_h']:5.1f}km/h  "
                    f"lat={obj['latitude']:.6f}  lon={obj['longitude']:.6f}  "
                    f"activity={obj['activity'] or '-'}"
                    f"body_lean={obj['body_lean'] or '-'}"
                    f"ttrc ={obj['ttrc'] or '-'}"
                )
            except queue.Empty:
                print("[TEST] Waiting for data …")
    except KeyboardInterrupt:
        pass
    finally:
        stop_pipeline()
        t.join(timeout=3)
        print("[TEST] Done.")