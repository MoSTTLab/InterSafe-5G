#!/usr/bin/env python3
"""
MEC Radar Server  v2
=====================
Runs on MEC server (Ubuntu).

Chain:
  Radar RS485 -> USB-RS485 -> RUTX50 (5G) -> [this script on MEC]

What this does:
  1. Listens on TCP port 5000 for RUTX50 serial-over-IP connection
  2. Decodes raw RS485 smartmicro frames (0x7E, CRC-16 CCITT)
  3. Decodes CAN messages: 0x0500, 0x0501, 0x0502..0x057F
  4. Classifies vehicles by length (per UMRR-11 firmware v4.3.2.1 spec)
  5. Detects and confirms pedestrians with zone filtering
  6. Logs heartbeat, objects, pedestrians to CSV
  7. Prints live display

Key fixes vs v1:
  - Correct bit-field extraction for object_data (matches spec §5.2.1 exactly)
  - Correct direction logic: heading is signed ±180°, approaching = |hdg| > 90°
  - Cycle state machine is frame-independent (status/control can span frames)
  - recv buffer enlarged to 4096 bytes to avoid frame fragmentation
  - Detection-zone filter now applied before logging too (was only for pedestrians)
  - Automatic reconnect: detector/logger state is preserved across reconnects
  - Pedestrian track age-out: old track_seen entries expire after N missed cycles
  - Thread-safe logger with a lock on CSV writes
  - Config validated at startup

Usage:
    python3 mec_radar_server_v2.py

RUTX50 Serial Over IP must be set to:
    Mode   : Client
    Host   : <this MEC server IP>
    Port   : 5000
"""
from collections import deque, defaultdict
import math
import sys
import os
import socket
import struct
import time
import csv
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple

latest_objects = []

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LISTEN_HOST        = "0.0.0.0"
LISTEN_PORT        = 5000           # Must match RUTX50 Serial Over IP port

LOG_DIR            = os.path.join(os.path.expanduser("~"), "mec_radar_logs")

# Detection zone (only objects inside this box are tracked / logged as "in zone")
MAX_DISTANCE_M     = 120.0          # radial distance from sensor
MAX_LATERAL_M      = 6.0            # |y| sideways limit

# Pedestrian confirmation: object must appear as pedestrian for this many
# consecutive sensor cycles before an alert is raised.
MIN_CONFIRM_CYCLES = 2

# If a pedestrian track disappears for more than this many cycles the counter resets.
TRACK_TIMEOUT_CYCLES = 5
# ==============================================================================
# TTRC CONFIG
# ==============================================================================

TTRC_THRESHOLD = 11.0      # seconds
TRACK_HISTORY_LEN = 20
MIN_SPEED_TTRC = 0.3       # m/s

# Example conflict point in radar coordinates
CONFLICT_POINTS = [
    (20.0, 0.0),
]

# Display
SHOW_ALL_OBJECTS   = True           # print every decoded object
SHOW_INZONE_ONLY   = False          # when True, only show objects inside the zone
SHOW_HEARTBEAT     = True
SHOW_EMPTY_CYCLES  = False
SUPPRESS_BLIND     = True           # hide "BLIND" from status line (sensor covers it)

# Recv buffer size. One RS485 frame with 64 objects ≈ 700 bytes; 4096 is safe.
RECV_BUF           = 4096

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("radar")


# ==============================================================================
# CRC-16 / CCITT-FALSE  (poly 0x1021, init 0xFFFF)
# ==============================================================================
def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


# ==============================================================================
# RS485 FRAME PARSER
# ==============================================================================
class RS485FrameParser:
    """
    Stateful parser for the Smartmicro RS485 transport protocol (§12).

    Feed raw bytes as they arrive; get back lists of (can_id, data) tuples
    extracted from complete, CRC-verified frames.
    """

    def __init__(self):
        self._buf            = bytearray()
        self.frames_ok       = 0
        self.frames_bad_crc  = 0
        self.frames_invalid  = 0

    def feed(self, data: bytes) -> List[List[Tuple[int, bytes]]]:
        """Append raw bytes and return all complete frames decoded so far."""
        self._buf.extend(data)
        return self._parse_all()

    def _parse_all(self) -> List[List[Tuple[int, bytes]]]:
        results = []
        while True:
            # Locate start byte 0x7E
            idx = self._buf.find(0x7E)
            if idx == -1:
                self._buf.clear()
                break
            if idx > 0:
                self._buf = self._buf[idx:]

            # Need at least header minimum (8 bytes) + 1 byte start = 9 visible bytes
            if len(self._buf) < 9:
                break

            # Byte 2 = header length (min 8, max 64 per §12.1.3)
            hdr_len = self._buf[2]
            if not (8 <= hdr_len <= 64):
                self._buf = self._buf[1:]
                self.frames_invalid += 1
                continue

            if len(self._buf) < hdr_len:
                break   # wait for more data

            # Bytes 3-4 = payload length (big-endian per §12)
            payload_len = struct.unpack_from('>H', self._buf, 3)[0]
            total       = hdr_len + payload_len + 2   # +2 for payload CRC

            if len(self._buf) < total:
                break   # wait for more data

            frame = bytes(self._buf[:total])

            # Verify header CRC (covers bytes 0 .. hdr_len-3, stored at hdr_len-2)
            hdr_crc_stored = struct.unpack_from('>H', frame, hdr_len - 2)[0]
            hdr_crc_calc   = crc16_ccitt(frame[:hdr_len - 2])
            if hdr_crc_stored != hdr_crc_calc:
                self._buf = self._buf[1:]
                self.frames_bad_crc += 1
                continue

            # Verify payload CRC
            payload        = frame[hdr_len: hdr_len + payload_len]
            pay_crc_stored = struct.unpack_from('>H', frame, hdr_len + payload_len)[0]
            pay_crc_calc   = crc16_ccitt(payload)
            if pay_crc_stored != pay_crc_calc:
                self._buf = self._buf[1:]
                self.frames_bad_crc += 1
                continue

            results.append(self._extract_can(payload))
            self.frames_ok += 1
            self._buf = self._buf[total:]

        return results

    @staticmethod
    def _extract_can(payload: bytes) -> List[Tuple[int, bytes]]:
        """
        Extract CAN messages from payload block (Table 12-3).
        Layout per message: [CAN_ID_HI][CAN_ID_LO][LEN][DATA...]
        """
        msgs, offset = [], 0
        while offset + 3 <= len(payload):
            can_id  = struct.unpack_from('>H', payload, offset)[0]
            can_len = payload[offset + 2]
            offset += 3
            if offset + can_len > len(payload):
                break
            msgs.append((can_id, bytes(payload[offset: offset + can_len])))
            offset += can_len
        return msgs


# ==============================================================================
# CAN MESSAGE DECODERS
# ==============================================================================

def decode_sensor_control(data: bytes) -> Optional[dict]:
    """
    Decode 0x0500 sensor_control message (Table 5-1).
    Format: Intel (little-endian).
      Byte 0      : Sensor_Status  (u8)
      Byte 1[0:3] : Interface_Mode (u4)
      Byte 1[4:7] : Network_ID     (u4)
      Byte 2      : Diagnose       (u8)
      Byte 3      : Reserve        (u8)
      Bytes 4-7   : Time           (u32, ms)
    """
    if len(data) < 8:
        return None
    return {
        'raw_status'  : data[0],
        'timestamp_ms': struct.unpack_from('<I', data, 4)[0],
    }


def decode_object_control(data: bytes) -> Optional[dict]:
    """
    Decode 0x0501 object_control message (Table 5-3).
    Format: Intel (little-endian).
      Byte 0      : Number_Of_Objects    (u8)
      Byte 1      : Number_Of_Messages   (u8)
      Byte 2      : Cycle_Duration       (u8, ms)
      Byte 3[0:3] : Object_data0_format  (u4)  — 7 = x/y/speed/heading
      Byte 3[4:7] : Object_data1_format  (u4)  — 1 = TM extension
      Bytes 4-7   : Cycle_Count          (u32)
    """
    if len(data) < 8:
        return None
    return {
        'num_objects'      : data[0],
        'cycle_duration_ms': data[2],
        'obj_data0_fmt'    : data[3] & 0x0F,
        'obj_data1_fmt'    : (data[3] >> 4) & 0x0F,
        'cycle_count'      : struct.unpack_from('<I', data, 4)[0],
    }


def decode_object_data(can_id: int, data: bytes) -> Optional[dict]:
    """
    Decode 0x0502..0x057F object_data_1 message (Table in §5.2.1).
    Format: Intel (little-endian), 64-bit word.

    Bit layout (LSB = bit 0):
      Bit  0       : Mode_Signal1  (u1)  — always 0, skip
      Bits 1..13   : x_Point1      (u13, offset 4096, resolution 0.128 m)
      Bits 14..26  : y_Point1      (u13, offset 4096, resolution 0.128 m)
      Bits 27..37  : Speed_Abs     (u11, offset 1024, resolution 0.1 m/s)
      Bits 38..48  : Heading       (u11, offset 1024, resolution 0.177 deg)
      Bits 49..55  : Object_Length (u7,  offset 0,    resolution 0.2 m)
      Bits 56..63  : Object_ID     (u8,  offset 0)
    """
    if not (0x0502 <= can_id <= 0x057F) or len(data) < 8:
        return None

    raw = int.from_bytes(data[:8], byteorder='little')

    # Extract each field with correct masks
    x_raw       = (raw >>  1) & 0x1FFF   # 13 bits
    y_raw       = (raw >> 14) & 0x1FFF   # 13 bits
    speed_raw   = (raw >> 27) & 0x7FF    # 11 bits
    heading_raw = (raw >> 38) & 0x7FF    # 11 bits
    length_raw  = (raw >> 49) & 0x7F     #  7 bits
    object_id   = (raw >> 56) & 0xFF     #  8 bits

    # Apply offset and resolution
    x_m     = (x_raw      - 4096) * 0.128   # signed metres
    y_m     = (y_raw      - 4096) * 0.128   # signed metres
    # Speed is always positive (absolute); clamp any decode artefact below 0
    spd_mps = max(0.0, (speed_raw - 1024) * 0.1)
    # Heading is signed ±180°; (raw - 1024) * 0.177 gives ~-181° .. +180°
    hdg_deg = (heading_raw - 1024) * 0.177
    len_m   = length_raw * 0.2
    dist_m  = (x_m ** 2 + y_m ** 2) ** 0.5

    return {
        "id"     : object_id,
        "x_m"    : round(x_m,    3),
        "y_m"    : round(y_m,    3),
        "dist_m" : round(dist_m, 3),
        "spd_mps": round(spd_mps, 3),
        "spd_kmh": round(spd_mps * 3.6, 2),
        "hdg_deg": round(hdg_deg, 2),
        "len_m"  : round(len_m,  1),
        "len_raw": length_raw,
    }


def flags_str(raw_status: int) -> str:
    """
    Decode Sensor_Status byte (Table 5-2).
    Bits 4..7 are Diagnostic bits 0..3 (Blind, Error, Interference, Rain).
    NOTE: The doc maps Diag bits to bits 4..7 of the status byte.
    """
    f = []
    if raw_status & 0x10: f.append("BLIND")
    if raw_status & 0x20: f.append("ERROR")
    if raw_status & 0x40: f.append("INTERFERENCE")
    if raw_status & 0x80: f.append("RAIN")
    return " | ".join(f) if f else "OK"


# ==============================================================================
# VEHICLE CLASSIFICATION  (per spec §5.2.1, UMRR-11 fw v4.3.2.1+)
# ==============================================================================
VEHICLE_CLASSES = [
    (0.0,  0.7,  "No object"),       # below pedestrian threshold
    (0.8,  1.2,  "Pedestrian"),      # fixed length 1.0 m → ±20% band
    (1.3,  1.9,  "Bicycle"),         # fixed length 1.6 m
    (2.0,  3.2,  "Two-wheeler"),       # fixed length 2.6 m
    (3.3,  5.6,  "Car"),   # 4.6 .. 5.4 m estimated
    (5.7,  9.2,  "Tempo-traveller"), # 5.6 .. 8.8 m estimated
    (9.3,  14.2, "Bus"),     # 9.0 .. 13.8 m estimated
    (14.3, 999,  "Heavy Vehicle"),      # >= 14.0 m estimated
]


def classify_vehicle(length_m: float) -> str:
    for lo, hi, label in VEHICLE_CLASSES:
        if lo <= length_m <= hi:
            return label
    return f"Unknown({length_m:.1f}m)"


# --------------------------------------------------------------------------
# TESTER-FACING CLASSIFICATION
# --------------------------------------------------------------------------
# Same length bands as VEHICLE_CLASSES above, but using the class-name
# strings tester_final.py expects in its TTRC1_CLASSES / VEHICLE_CLASSES
# sets ("Car" / "Transporter" instead of "Passenger Car" / "Delivery/Pickup").
# Computed here so tester_final.py no longer needs its own classifier and
# can just read obj["class"] straight off latest_objects.
TESTER_VEHICLE_CLASSES = [
    (0.0,  0.7,  "No object"),
    (0.8,  1.2,  "Pedestrian"),
    (1.3,  1.9,  "Bicycle"),
    (2.0,  3.2,  "Two-wheeler"),
    (3.3,  5.6,  "Car"),
    (5.7,  9.2,  "Tempo-traveller"),
    (9.3,  14.2, "Bus"),
    (14.3, 999,  "Heavy Vehicle"),
]


def classify_for_tester(length_m: float) -> str:
    for lo, hi, label in TESTER_VEHICLE_CLASSES:
        if lo <= length_m <= hi:
            return label
    return f"Unknown({length_m:.1f}m)"


def is_pedestrian(length_m: float) -> bool:
    return 0.8 <= length_m <= 1.2


def direction_str(hdg_deg: float) -> str:
    """
    Heading convention (spec §5.2.1):
      hdg_deg is signed ±180°.
      Objects moving toward the sensor have heading ~±180° (large |hdg|).
      Objects moving away have heading ~0° (small |hdg|).
    Threshold: if |hdg| > 90° → APPROACHING, else RECEDING.
    """
    return "APPROACHING" if abs(hdg_deg) > 90.0 else "RECEDING"


def in_zone(obj: dict) -> bool:
    """Return True if this object is inside the configured detection zone."""
    return obj['dist_m'] <= MAX_DISTANCE_M and abs(obj['y_m']) <= MAX_LATERAL_M

# ==============================================================================
# TTRC TRACKER
# ==============================================================================

class TTRCTracker:

    def __init__(self):

        self.history = defaultdict(
            lambda: deque(maxlen=TRACK_HISTORY_LEN)
        )

        self.ttrc_per_id = {}

    def update(self, obj, cycle_num, cycle_duration_ms):

        oid = obj["id"]

        self.history[oid].append(
            (
                cycle_num,
                obj["x_m"],
                obj["y_m"]
            )
        )

        hist = self.history[oid]

        if len(hist) < 2:
            return None

        cycle_old, x_old, y_old = hist[0]
        cycle_new, x_new, y_new = hist[-1]

        dt = ((cycle_new - cycle_old) * cycle_duration_ms) / 1000.0

        if dt <= 0:
            return None

        dx = x_new - x_old
        dy = y_new - y_old

        speed = math.hypot(dx, dy) / dt

        if speed < MIN_SPEED_TTRC:
            return None


        for cp_x, cp_y in CONFLICT_POINTS:

            dx_cp = cp_x - x_new
            dy_cp = cp_y - y_new

            dot = dx * dx_cp + dy * dy_cp

            if dot <= 0:
                continue

            dist_cp = math.hypot(dx_cp, dy_cp)

            ttrc = dist_cp / speed

        if ttrc is not None:
            self.ttrc_per_id[oid] = round(ttrc, 3)

        return ttrc
# ==============================================================================
# PEDESTRIAN DETECTOR
# ==============================================================================
class PedestrianDetector:
    """
    Confirms a pedestrian only after it appears in MIN_CONFIRM_CYCLES consecutive
    (or near-consecutive) sensor cycles within the detection zone.

    A track is aged out after TRACK_TIMEOUT_CYCLES cycles without being seen.
    """

    def __init__(self):
        # {object_id: consecutive_seen_count}
        self._track_seen     : Dict[int, int] = {}
        # {object_id: cycles_since_last_seen}
        self._track_missing  : Dict[int, int] = {}
        # IDs currently in confirmed/active state (prevents duplicate alerts)
        self._active_ids     : set            = set()
        self.confirmed_alerts: int            = 0
        self.length_histogram: Dict[int, int] = {}

    def process(self, objects: list, rx_time: float = None) -> list:
        """
        Call once per sensor cycle.
        Returns list of objects that are confirmed pedestrians.
        """
        current_ped_ids: set  = set()
        confirmed_peds : list = []
        global latest_objects

        if rx_time is None:
            rx_time = time.time()

        latest_objects = []

        for o in objects:
            latest_objects.append({
                "id": o["id"],
                "x": o["x_m"],
                "y": o["y_m"],
                "speed": o["spd_kmh"],
                "heading": o["hdg_deg"],
                "len_m": o["len_m"],
                "class": classify_for_tester(o["len_m"]),  # class label computed here, not in tester_final.py
                "timestamp": rx_time,  # radar rx timestamp (epoch seconds), computed here not in tester_final.py
                "ttrc": o.get("ttrc")  # seconds, or None if not computable yet
                })

        for obj in objects:
            # Update length histogram (all objects, for diagnostic)
            key = obj['len_raw']
            self.length_histogram[key] = self.length_histogram.get(key, 0) + 1

            # Only consider pedestrians inside the zone
            if not is_pedestrian(obj['len_m']) or not in_zone(obj):
                continue

            oid = obj['id']
            current_ped_ids.add(oid)

            # Increment seen counter; reset missing counter
            self._track_seen[oid]    = self._track_seen.get(oid, 0) + 1
            self._track_missing[oid] = 0

            if self._track_seen[oid] >= MIN_CONFIRM_CYCLES:
                confirmed_peds.append(obj)
                if oid not in self._active_ids:
                    self.confirmed_alerts += 1
                self._active_ids.add(oid)

        # Age out tracks that were not seen this cycle
        to_delete = []
        for oid in list(self._track_missing.keys()):
            if oid not in current_ped_ids:
                self._track_missing[oid] = self._track_missing.get(oid, 0) + 1
                if self._track_missing[oid] > TRACK_TIMEOUT_CYCLES:
                    to_delete.append(oid)

        # Also start missing counter for brand-new tracks not seen this cycle
        for oid in list(self._track_seen.keys()):
            if oid not in current_ped_ids and oid not in self._track_missing:
                self._track_missing[oid] = 1

        for oid in to_delete:
            self._track_seen.pop(oid, None)
            self._track_missing.pop(oid, None)
            self._active_ids.discard(oid)

        return confirmed_peds

    def print_histogram(self):
        if not self.length_histogram:
            return
        print("\n  length_raw histogram:")
        print(f"  {'raw':>4}  {'len_m':>6}  {'count':>7}  {'class':<17}")
        print("  " + "-" * 42)
        for raw in sorted(self.length_histogram):
            print(f"  {raw:4d}  {raw * 0.2:6.1f}m  "
                  f"{self.length_histogram[raw]:7d}  "
                  f"[{classify_vehicle(raw * 0.2):<15s}]")


# ==============================================================================
# CSV LOGGER  (thread-safe)
# ==============================================================================
class DataLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hb_path  = os.path.join(log_dir, f"hb_{ts}.csv")
        self.obj_path = os.path.join(log_dir, f"objects_{ts}.csv")
        self.ped_path = os.path.join(log_dir, f"pedestrians_{ts}.csv")

        self._lock  = threading.Lock()

        self._hb_f  = open(self.hb_path,  'w', newline='', encoding='utf-8')
        self._obj_f = open(self.obj_path, 'w', newline='', encoding='utf-8')
        self._ped_f = open(self.ped_path, 'w', newline='', encoding='utf-8')

        self._hb_w  = csv.writer(self._hb_f)
        self._obj_w = csv.writer(self._obj_f)
        self._ped_w = csv.writer(self._ped_f)

        self._hb_w.writerow([
            'rx_time', 'cycle', 'dur_ms', 'uptime_ms',
            'num_objects', 'status'
        ])
        self._obj_w.writerow([
            'rx_time',
            'cycle',
            'object_id',
            'vehicle_class',
            'x_m',
            'y_m',
            'dist_m',
            'spd_mps',
            'spd_kmh',
            'hdg_deg',
            'direction',
            'len_m',
            'len_raw',
            'in_zone',
            'ttrc_sec'
        ])
        self._ped_w.writerow([
            'rx_time', 'cycle', 'object_id',
            'x_m', 'y_m', 'dist_m',
            'spd_mps', 'spd_kmh',
            'hdg_deg', 'direction',
            'len_m', 'len_raw'
        ])

    def log_heartbeat(self, cycle: int, dur_ms: int, uptime_ms: int,
                      num_objects: int, status: str, rx_time: float):
        with self._lock:
            self._hb_w.writerow([
                f"{rx_time:.3f}", cycle, dur_ms, uptime_ms, num_objects, status
            ])
            self._hb_f.flush()

    def log_object(self, obj: dict, cycle: int, rx_time: float,ttrc=None):
        with self._lock:
            self._obj_w.writerow([
                f"{rx_time:.3f}", cycle, obj['id'],
                classify_vehicle(obj['len_m']),
                f"{obj['x_m']:.3f}", f"{obj['y_m']:.3f}", f"{obj['dist_m']:.3f}",
                f"{obj['spd_mps']:.3f}", f"{obj['spd_kmh']:.2f}",
                f"{obj['hdg_deg']:.2f}", direction_str(obj['hdg_deg']),
                f"{obj['len_m']:.1f}", obj['len_raw'],
                1 if in_zone(obj) else 0 , ttrc
            ])
            self._obj_f.flush()

    def log_pedestrian(self, obj: dict, cycle: int, rx_time: float):
        with self._lock:
            self._ped_w.writerow([
                f"{rx_time:.3f}", cycle, obj['id'],
                f"{obj['x_m']:.3f}", f"{obj['y_m']:.3f}", f"{obj['dist_m']:.3f}",
                f"{obj['spd_mps']:.3f}", f"{obj['spd_kmh']:.2f}",
                f"{obj['hdg_deg']:.2f}", direction_str(obj['hdg_deg']),
                f"{obj['len_m']:.1f}", obj['len_raw']
            ])
            self._ped_f.flush()

    def close(self):
        with self._lock:
            for f in [self._hb_f, self._obj_f, self._ped_f]:
                try:
                    f.close()
                except Exception:
                    pass


# ==============================================================================
# CLIENT HANDLER
# ==============================================================================
def handle_client(conn: socket.socket, addr: tuple,
                  ped_det: PedestrianDetector, logger: DataLogger, ttrc_tracker):
    """
    Handle one RUTX50 TCP connection.

    Cycle detection is stateful and frame-independent:
    - 0x0500 sets pending_status
    - 0x0501 sets pending_control and resets the object list
    - 0x0502+ appends to pending_objects
    - A complete cycle is emitted whenever BOTH pending_status AND
      pending_control are populated and we see the start of a NEW 0x0501
      (or at end of each RS485 frame if both are present).

    This matches the sensor's output pattern: 0x0500 → 0x0501 → 0x0502..0x05xx
    all in one RS485 frame per cycle.
    """
    log.info(f"[TCP] RUTX50 connected from {addr[0]}:{addr[1]}")

    parser           = RS485FrameParser()
    pending_status   : Optional[dict] = None
    pending_control  : Optional[dict] = None
    pending_objects  : list           = []
    cycles           = 0
    t_start          = time.time()
    last_stat_time   = time.time()

    def emit_cycle():
        nonlocal cycles
        if pending_status is None or pending_control is None:
            return

        rx_time   = time.time()
        cycle_num = pending_control['cycle_count']
        dur_ms    = pending_control['cycle_duration_ms']
        uptime_ms = pending_status['timestamp_ms']
        raw_st    = pending_status['raw_status']
        status    = flags_str(raw_st)
        objects   = list(pending_objects)
        cycles   += 1

        # Strip BLIND from display if configured
        disp_status = status
        if SUPPRESS_BLIND:
            disp_status = (disp_status
                           .replace("BLIND | ", "")
                           .replace("BLIND", "")
                           .strip(" |") or "OK")

        # Log heartbeat
        logger.log_heartbeat(cycle_num, dur_ms, uptime_ms,
                             len(objects), status, rx_time)

        # Heartbeat print
        if SHOW_HEARTBEAT and (objects or SHOW_EMPTY_CYCLES):
            print(f"[HB] cycle={cycle_num:6d}  "
                  f"obj={len(objects):3d}  "
                  f"status={disp_status}  "
                  f"frames_ok={parser.frames_ok}")

        # Per-object display and logging
        for obj in objects:
            veh_class = classify_vehicle(obj['len_m'])
            direction = direction_str(obj['hdg_deg'])
            zone_flag = in_zone(obj)
            ped_flag  = "  *** PEDESTRIAN ***" if is_pedestrian(obj['len_m']) else ""
            zone_tag  = "" if zone_flag else "  [OUT-OF-ZONE]"
            ttrc = ttrc_tracker.update(obj,cycle_num,dur_ms)
            obj['ttrc'] = ttrc  # stash on the object so it flows into latest_objects too; threshold check + alerting happens in tester.py

            if SHOW_ALL_OBJECTS and (zone_flag or not SHOW_INZONE_ONLY):
                print(f"  [{veh_class:15s}] "
                      f"id={obj['id']:3d} "
                      f"dist={obj['dist_m']:6.1f}m "
                      f"x={obj['x_m']:7.2f}m "
                      f"y={obj['y_m']:6.2f}m "
                      f"spd={obj['spd_kmh']:5.1f}km/h "
                      f"hdg={obj['hdg_deg']:6.1f}° "
                      f"len={obj['len_m']:.1f}m(r={obj['len_raw']:3d}) "
                      f"{direction}{ped_flag}{zone_tag}")

            logger.log_object(obj, cycle_num, rx_time,ttrc)

        # Pedestrian detection (only in-zone candidates are processed inside)
        confirmed = ped_det.process(objects, rx_time)
        for ped in confirmed:
            direction = direction_str(ped['hdg_deg'])
            print(f"\n  *** PEDESTRIAN CONFIRMED ***")
            print(f"     Track ID  : {ped['id']}")
            print(f"     Distance  : {ped['dist_m']:.1f} m")
            print(f"     Position  : x={ped['x_m']:.2f} m   y={ped['y_m']:.2f} m")
            print(f"     Speed     : {ped['spd_kmh']:.1f} km/h")
            print(f"     Direction : {direction}")
            print(f"     Heading   : {ped['hdg_deg']:.1f}°")
            print(f"     Length    : {ped['len_m']:.1f} m  (raw={ped['len_raw']})")
            print()
            logger.log_pedestrian(ped, cycle_num, rx_time)

        # Periodic stats
        now = time.time()
        nonlocal last_stat_time
        if now - last_stat_time >= 10.0:
            last_stat_time = now
            runtime = now - t_start
            print(f"[STAT] runtime={runtime:.0f}s  "
                  f"cycles={cycles}  "
                  f"frames_ok={parser.frames_ok}  "
                  f"bad_crc={parser.frames_bad_crc}  "
                  f"invalid={parser.frames_invalid}  "
                  f"ped_confirmed={ped_det.confirmed_alerts}  "
                  f"from={addr[0]}")

    try:
        while True:
            chunk = conn.recv(RECV_BUF)
            if not chunk:
                log.info(f"[TCP] RUTX50 {addr[0]} disconnected (EOF).")
                break

            for can_messages in parser.feed(chunk):
                for can_id, can_data in can_messages:

                    if can_id == 0x0500:
                        d = decode_sensor_control(can_data)
                        if d:
                            pending_status = d

                    elif can_id == 0x0501:
                        # New cycle header: emit the previous cycle first
                        emit_cycle()
                        d = decode_object_control(can_data)
                        if d:
                            pending_control = d
                            pending_objects = []

                    elif 0x0502 <= can_id <= 0x057F:
                        obj = decode_object_data(can_id, can_data)
                        if obj:
                            pending_objects.append(obj)

                # Emit at end of each RS485 frame if we have a complete cycle
                # This handles the common case where 0x0500+0x0501+objects
                # arrive in the same frame without a following 0x0501.
                if pending_status is not None and pending_control is not None:
                    emit_cycle()
                    pending_status  = None
                    pending_control = None
                    pending_objects = []

    except ConnectionResetError:
        log.warning(f"[TCP] {addr[0]} connection reset.")
    except OSError as e:
        log.error(f"[TCP] {addr[0]} OS error: {e}")
    except Exception as e:
        log.exception(f"[ERROR] {addr[0]}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        log.info(f"[TCP] Connection from {addr[0]} closed. "
                 f"cycles={cycles}  "
                 f"frames_ok={parser.frames_ok}  "
                 f"bad_crc={parser.frames_bad_crc}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 65)
    print("  UMRR-11 Radar  —  MEC Server  v2")
    print(f"  Listening on  : {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"  Logs dir      : {LOG_DIR}")
    print(f"  Zone          : dist<={MAX_DISTANCE_M}m  |y|<={MAX_LATERAL_M}m")
    print(f"  Ped confirm   : {MIN_CONFIRM_CYCLES} cycles  "
          f"(timeout {TRACK_TIMEOUT_CYCLES} cycles)")
    print("=" * 65)
    print("  Waiting for RUTX50 connection...")
    print("  (RUTX50: Serial Over IP → Mode=Client, "
          f"Host=<MEC IP>, Port={LISTEN_PORT})")
    print("=" * 65 + "\n")

    ped_det = PedestrianDetector()
    ttrc_tracker = TTRCTracker()
    logger  = DataLogger(LOG_DIR)

    log.info(f"HB  log -> {logger.hb_path}")
    log.info(f"Obj log -> {logger.obj_path}")
    log.info(f"Ped log -> {logger.ped_path}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(5)

    try:
        while True:
            try:
                conn, addr = server.accept()
                conn.settimeout(30.0)
                t = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, ped_det, logger, ttrc_tracker),
                    daemon=True,
                    name=f"radar-{addr[0]}",
                )
                t.start()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.error(f"[ERROR] Accept: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        print(f"[INFO] Total pedestrian alerts: {ped_det.confirmed_alerts}")
        ped_det.print_histogram()

    finally:
        server.close()
        logger.close()
        log.info("Server closed. Logs saved.")


if __name__ == "__main__":
    main()