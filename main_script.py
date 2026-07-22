"""
main_script.py  —  InterSafe-5G  Unified Entry Point
=====================================================
Run this file to start the entire system:

    python3 main_script.py --source RADAR
    python3 main_script.py --source CAMERA
    python3 main_script.py --source BOTH

SYSTEM TOPOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Radar (UMRR-11)
    └─RS485─► 5G gateway (RUTX50) ─TCP 5000─► radar_data.py (MEC)

  IP Camera (RTSP)
    └─────────────────────────────────────────► camera_data.py (MEC)

  main_script.py (MEC)
    ├── reads radar_data.latest_objects / camera_data.track_queue
    ├── applies TTRC thresholds
    ├── calls alert_dispatcher.send(alert)
    │     ├──► VMS board (TCP via gateway port-forward)
    │     └──► Mobile UE (CoAP Observe via coap_server.py)
    └── logs alerts to CSV
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Startup order
─────────────
  1. CoAP server starts (background thread + event loop)
  2. alert_dispatcher.init()  — VMS board + CoAP client ready
  3. VMS → GREEN OK  (idle state)
  4. Sensor pipelines launch (radar / camera / both)
  5. Interactive source-switcher thread starts
  6. Main loop polls sensors at 100 ms, fires alerts via alert_dispatcher

What lives here
───────────────
  ✔ TTRC threshold logic
  ✔ Sensor-reading loops (radar + camera)
  ✔ Alert message formatting
====================================================  
"""

import argparse
import csv
import os
import threading
import time
from datetime import datetime

import radar_data
import camera_data
import coap_server                  # ← CoAP server: /alert, /gps, /heartbeat
import alert_dispatcher         # ← handles all delivery to VMS + phone

# =============================================================================
# CONFIG
# =============================================================================

COAP_PORT           = coap_server.COAP_PORT   # kept here only for the startup banner

TTRC1_THRESHOLD     = 10.0      # Pedestrian / Bicycle  (s)
TTRC2_THRESHOLD     = 15.0      # All vehicle classes   (s)
TTRC1_CLASSES       = {"Pedestrian", "Bicycle"}

GROUP_ALERT_COUNT = 2      # group-alert threshold

# =============================================================================
# CSV ALERT LOG CONFIG
# =============================================================================

CSV_OUTPUT_DIR = ""   # change to your path
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

_session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH    = os.path.join(CSV_OUTPUT_DIR, f"alerts_{_session_ts}.csv")

CSV_COLUMNS = ["timestamp", "source", "class", "obj_id", "ttrc",
               "ttrc_tier", "speed_kmh", "status", "direction", "move_dir", "alert_message"]

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description="InterSafe-5G Unified Entry Point")
parser.add_argument(
    "--source", choices=["RADAR", "CAMERA", "BOTH"], default="RADAR",
    help="Active sensor source (default: RADAR)"
)
args        = parser.parse_args()
DATA_SOURCE = args.source

# =============================================================================
# HELPERS
# =============================================================================

def _ttrc_threshold(label: str) -> float:
    return TTRC1_THRESHOLD if label in TTRC1_CLASSES else TTRC2_THRESHOLD


def _ttrc_tier(label: str) -> str:
    return "TTRC1" if label in TTRC1_CLASSES else "TTRC2"


# =============================================================================
# ALERT MESSAGE FORMATTER
# =============================================================================
def _format_alert_message(label: str, track_id, ttrc_s: float,
                           ids_in_frame: list,
                           body_lean: str | None = None,
                           move_dir: str = "") -> str:
    """
    Build the human-readable alert string for a single object.

    ids_in_frame — list of IDs of ALL objects of the same class currently
                   in the frame (used for group-alert detection).
                   If more than GROUP_ALERT_COUNT objects of this class are
                   present, the message switches to "Multiple <Class> AHEAD"
                   regardless of class.  This applies to every class, not
                   just pedestrians.

    body_lean    — only populated for Pedestrians via pose estimation.
    """
    tier     = _ttrc_tier(label)
    ttrc_str = f"{ttrc_s:.1f}s [{tier}]"

    # ── Group alert: more than GROUP_ALERT_COUNT of this class in frame ───────
    if len(ids_in_frame) > GROUP_ALERT_COUNT:
        # Pluralise the label for the message
        plural = {
            "Pedestrian":    "Pedestrians",
            "Bicycle":       "Bicycles",
            "Two-wheeler":   "Two-wheelers",
            "Three-wheeler": "Three-wheelers",
            "Car":           "Cars",
            "Tempo-Traveller":"Tempo-Travellers",
            "Bus":           "Buses",
            "Heavy Vehicle": "Heavy Vehicles",
        }.get(label, f"{label}s")
        return f"{plural} AHEAD | {ttrc_str}"

    # ── Single object alert ───────────────────────────────────────────────────
    msg = f"{label} ID:{track_id} AHEAD | {ttrc_str}"

    if label == "Pedestrian":
        # Pedestrians: use pose body_lean (more accurate than px-based move_dir)
        if body_lean and body_lean != "straight":
            msg += f" | moving {body_lean}"
    else:
        # Vehicles: use px-based lateral direction
        msg += f" | moving {move_dir}"
    """        
    else:
        # Vehicles: use px-based lateral direction
        if move_dir and move_dir != "straight":
            msg += f" | moving {move_dir}"
    """
    return msg
# =============================================================================
# EMIT ALERT  (build dict, print, hand off to dispatcher)
# =============================================================================

def emit_alert(source: str, label: str, object_id: int,
               ttrc: float, speed_kmh: float,
               direction: str = "", timestamp: float | None = None,
               ids_in_frame: list | None = None,
               body_lean: str | None = None,
               move_dir: str = "") -> None:
    """
    Build the alert payload, print to terminal, and hand off to
    alert_dispatcher.send() for delivery to VMS + phone.

    This function contains NO delivery code itself — that belongs entirely
    in alert_dispatcher.  Adding a new channel (SMS, Slack, syslog …)
    means editing alert_dispatcher only.
    """
    ids_in_frame = ids_in_frame or []
    tier             = _ttrc_tier(label)
    status           = "Reached CP" if ttrc < 0.2 else "Reaching CP"
    if timestamp is None:
        timestamp = time.time()

    alert_message = _format_alert_message(
        label, object_id, ttrc, ids_in_frame, body_lean, move_dir
    )

    alert = {
        "type":             "CONFLICT",
        "source":           source.lower(),
        "class":            label,
        "object_id":        object_id,
        "ttrc":             round(ttrc,      2),
        "ttrc_tier":        tier,
        "speed_kmh":        round(speed_kmh, 1),
        "status":           status,
        "direction":        direction,
        "move_dir":         move_dir,
        "timestamp":        timestamp,
        "alert_message":    alert_message,
        "ids_in_frame": ids_in_frame,  # forwarded to dispatcher for VMS grouping
    }

    # ── CSV log ──────────────────────────────────────────────────────────────
    with _csv_lock:
        _csv_writer.writerow({
            "wall_time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "source":        source,
            "class":         label,
            "obj_id":        object_id,
            "ttrc":          round(ttrc, 2),
            "ttrc_tier":     tier,
            "speed_kmh":     round(speed_kmh, 1),
            "status":        status,
            "direction":     direction,
            "move_dir":      move_dir,
            "alert_message": alert_message,
        })
        _csv_file.flush()

    # ── Terminal print ────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"  ⚠️   CONFLICT ALERT  [{source}]")
    print("=" * 52)
    print(f"  Obj ID     : {object_id}")
    print(f"  Class      : {label}")
    print(f"  Message    : {alert_message}")
    print(f"  TTRC       : {ttrc:.2f}s  [{tier}]  (threshold {_ttrc_threshold(label):.0f}s)")
    print(f"  Speed      : {speed_kmh:.1f} km/h")
    print(f"  Status     : {status}")
    if direction:
        print(f"  Direction  : {direction}")
    print(f"  Timestamp  : {timestamp:.3f}")
    print("=" * 52)

    # ── Deliver (VMS + phone) — all channel logic lives in alert_dispatcher ──
    alert_dispatcher.send(alert)

# =============================================================================
# RADAR PROCESSING
# =============================================================================

_radar_alerted: set = set()

def process_radar():
    objects = radar_data.latest_objects
    if not objects:
        return

    for obj in objects:
        ttrc      = obj.get("ttrc")
        oid       = obj["id"]
        label     = obj.get("class", "Unknown")
        speed_kmh = obj.get("spd_kmh", obj.get("speed", 0.0))
        direction = "APPROACHING" if abs(obj.get("hdg_deg", obj.get("heading", 0))) > 90 else "RECEDING"
        timestamp = obj.get("timestamp")
        threshold = _ttrc_threshold(label)

        if ttrc is None or ttrc > threshold:
            continue

        tier = _ttrc_tier(label)
        print(f"[RADAR] ID:{oid:3d}  {label:15s}  "
              f"TTRC={ttrc:6.2f}s  [{tier}]  "
              f"spd={speed_kmh:5.1f}km/h  {direction}")

        if oid not in _radar_alerted:
            _radar_alerted.add(oid)
            same_class_ids = [o["id"] for o in objects if o.get("class") == label]
            emit_alert(
                source           = "RADAR",
                label            = label,
                object_id        = oid,
                ttrc             = ttrc,
                speed_kmh        = speed_kmh,
                direction        = direction,
                timestamp        = timestamp,
                ids_in_frame = same_class_ids,
            )

# =============================================================================
# CAMERA PROCESSING
# =============================================================================

_camera_alerted: set = set()


def process_camera():
    # Drain the full queue every cycle so no frame backlog builds up
    items = []
    while not camera_data.track_queue.empty():
        try:        
            items.append(camera_data.track_queue.get_nowait())
        except Exception:
            break

    if DATA_SOURCE not in ("CAMERA", "BOTH") or not items:
        return

    for obj in items:
        ttrc = obj.get("ttrc")
        if ttrc is None:
            continue

        label     = obj.get("label",      "Unknown")
        oid       = obj.get("obj_id",     -1)
        frame_no  = obj.get("frame_no",    0)
        speed_kmh = obj.get("speed_kmh", obj.get("speed_m_s", 0.0) * 3.6)
        move_dir  = obj.get("move_dir",   "")
        direction = obj.get("direction", "")
        body_lean = obj.get("body_lean",   None)
        timestamp = obj.get("timestamp")
        threshold = _ttrc_threshold(label)

        if ttrc > threshold:
            continue

        tier = _ttrc_tier(label)
        print(f"[CAMERA] Frame#{frame_no:05d}  ID:{oid:3d}  {label:15s}  "
              f"TTRC={ttrc:6.2f}s  [{tier}]  "
              f"spd={speed_kmh:5.1f}km/h")

        if oid not in _camera_alerted:
            _camera_alerted.add(oid)
            same_class_ids = [it.get("obj_id", -1) for it in items if it.get("label") == label]
            emit_alert(
                source           = "CAMERA",
                label            = label,
                object_id        = oid,
                ttrc             = ttrc,
                speed_kmh        = speed_kmh,
                direction        = direction,
                move_dir         = move_dir,
                timestamp        = timestamp,
                ids_in_frame = same_class_ids,
                body_lean        = body_lean,
            )

# =============================================================================
# INTERACTIVE SOURCE SWITCHER
# =============================================================================

def _command_listener():
    global DATA_SOURCE, _radar_alerted, _camera_alerted
    while True:
        cmd = input("\nEnter source (RADAR/CAMERA/BOTH): ").strip().upper()
        if cmd in ("RADAR", "CAMERA", "BOTH"):
            DATA_SOURCE     = cmd
            _radar_alerted  = set()
            _camera_alerted = set()
            print(f"[INFO] Switched to {DATA_SOURCE} — alert guards reset")
        else:
            print("Invalid — use RADAR, CAMERA, or BOTH")

# =============================================================================
# STARTUP
# =============================================================================

print("=" * 60)
print("  InterSafe-5G  —  Unified Entry Point")
print(f"  Source  : {DATA_SOURCE}")
print(f"  TTRC1   : <= {TTRC1_THRESHOLD}s  (Pedestrian / Bicycle)")
print(f"  TTRC2   : <= {TTRC2_THRESHOLD}s  (Vehicles)")
print(f"  VMS     : ERR + text for 3s, then GREEN OK")
print(f"  CoAP    : udp://0.0.0.0:{COAP_PORT}")
print("=" * 60)

# ── 0. CSV alert log ──────────────────────────────────────────────────────────
_csv_lock   = threading.Lock()
_csv_file   = open(CSV_PATH, "w", newline="")
_csv_writer = csv.DictWriter(_csv_file, fieldnames=CSV_COLUMNS)
_csv_writer.writeheader()
print(f"[CSV] Alert log → {CSV_PATH}")

# ── 1. CoAP server thread ─────────────────────────────────────────────────────
_coap_loop = coap_server.start()

# ── 2. Alert dispatcher (VMS board + CoAP client) ────────────────────────────
alert_dispatcher.init(_coap_loop)

# ── 3. VMS → GREEN OK (idle state) ───────────────────────────────────────────
print("[VMS] Initialising board → GREEN OK ...")
alert_dispatcher.set_vms_idle()

# ── 4. Sensor pipelines ───────────────────────────────────────────────────────
if DATA_SOURCE in ("RADAR", "BOTH"):
    threading.Thread(
        target=radar_data.main, daemon=True, name="radar-pipeline"
    ).start()
    print("[INFO] Radar pipeline started")

if DATA_SOURCE in ("CAMERA", "BOTH"):
    camera_data.start_pipeline()
    print("[INFO] Camera pipeline started")

# ── 5. Interactive switcher ────────────────────────────────────────────────────
threading.Thread(
    target=_command_listener, daemon=True, name="cmd-listener"
).start()

print("[INFO] All systems running. Press Ctrl-C to stop.\n")

# =============================================================================
# MAIN LOOP  — 100 ms poll cycle, runs both sensor processors in parallel
# =============================================================================

try:
    while True:
        if DATA_SOURCE in ("RADAR", "BOTH"):
            process_radar()

        if DATA_SOURCE in ("CAMERA", "BOTH"):
            process_camera()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")
    alert_dispatcher.shutdown()
    alert_dispatcher.set_vms_idle()
    _csv_file.close()
    print(f"[CSV] Alert log saved → {CSV_PATH}")
    print("[INFO] VMS → GREEN OK. Done.")