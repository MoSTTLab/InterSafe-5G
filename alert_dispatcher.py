"""
alert_dispatcher.py  —  InterSafe-5G  Alert Delivery Module
=============================================================
Delivers conflict alerts to two channels:
  1. VMS board  — UFD protocol over TCP (via 5G gateway port-forward)
  2. Mobile app — CoAP Observe (via aiocoap in-process POST)

NETWORK TOPOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MEC Server (this process)
  ├── alert_dispatcher   ──TCP GATEWAY_WAN_IP:5000──►  5G gateway
  │   (UFD commands)                                    (CPE / RUT)
  │                                                     └─LAN─► VMS board 192.168.1.75:5000
  │
  └── alert_dispatcher   ──CoAP POST 127.0.0.1:5683──► coap_server (same process)
      (aiocoap client)                                   └─Observe notify──► Mobile UE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import json
import threading
import time

import aiocoap

from vms_display_live import UFDBoard, UFD_TARGET_IP, UFD_TARGET_PORT

# =============================================================================
# CONFIG
# =============================================================================

COAP_SERVER_HOST = "127.0.0.1"
COAP_PORT              = 5683
VMS_ALERT_DURATION_SEC = 3.0    # how long VMS shows the alert before reverting

# =============================================================================
# VMS BOARD  (one shared instance)
# =============================================================================

_vms_board: UFDBoard = UFDBoard(UFD_TARGET_IP, UFD_TARGET_PORT)

def _vms_lines_for(label: str, track_id, source: str,
                   ped_ids_in_frame: list) -> tuple[str, str]:
    """Map alert label → two VMS display lines (≤12 chars each for readability)."""
    tag = "C" if source.upper() == "CAMERA" else "R"

    mapping = {
        "Pedestrian":    ("PEDESTRIAN",  f"AHEAD {tag}"),
        "Bicycle":       ("BICYCLE",     f"AHEAD {tag}"),
        "Two-wheeler":   ("TWO WHEELER", f"AHEAD {tag}"),
        "Three-wheeler": ("THREE WHEEL", f"AHEAD {tag}"),
        "Car":           ("CAR",         f"AHEAD {tag}"),
        "Tempo-traveler": ("TEMPO TRAVELLER", f"AHEAD {tag}"),
        "Bus":           ("BUS",         f"AHEAD {tag}"),
        "Heavy Vehicle": ("HEAVY VEH",   f"AHEAD {tag}"),
    }

    if label == "Pedestrian" and len(ped_ids_in_frame) > 2:
        return "PEDESTRIANS", f"AHEAD {tag}"

    line1, line2 = mapping.get(label, (label.upper()[:12], f"AHEAD {tag}"))
    return line1, line2

_vms_lock = threading.Lock()
def _trigger_vms(label: str, track_id, source: str,
                 ped_ids_in_frame: list) -> None:

    line1, line2 = _vms_lines_for(label, track_id, source, ped_ids_in_frame)

    def _run():
        print(f"[VMS] Static → '{line1} / {line2}'  ({VMS_ALERT_DURATION_SEC}s)")
        with _vms_lock:   # ← serialise all VMS operations
            _vms_board.clear_all()
            time.sleep(0.1)
            _vms_board.display_message(line1, line2, image="er")
            time.sleep(VMS_ALERT_DURATION_SEC)
            _vms_board.clear_all()
            time.sleep(0.1)
            _vms_board.show_ok_image()
        print("[VMS] Cleared → GREEN OK")

    threading.Thread(target=_run, daemon=True).start()


def set_vms_idle() -> None:
    _vms_board.show_ok_image()

# =============================================================================
# CoAP CLIENT  (posts alerts into the CoAP server loop → observers notified)
# =============================================================================

_coap_loop:       asyncio.AbstractEventLoop | None = None
_coap_client_ctx: aiocoap.Context | None           = None

async def _init_client() -> None:
    global _coap_client_ctx
    _coap_client_ctx = await aiocoap.Context.create_client_context()

async def _post_coap(payload: bytes) -> None:
    if _coap_client_ctx is None:
        print("[ALERT] CoAP client not ready — skipping phone notification")
        return
    try:
        req = aiocoap.Message(
            code    = aiocoap.POST,
            uri     = f"coap://{COAP_SERVER_HOST}:{COAP_PORT}/alert",
            payload = payload,
        )
        req.opt.content_format = 50          # application/json
        resp = await asyncio.wait_for(
            _coap_client_ctx.request(req).response, timeout=3.0
        )
        print(f"[CoAP] /alert → {resp.code}")
    except asyncio.TimeoutError:
        print("[CoAP] /alert POST timed out")
    except Exception as exc:
        print(f"[CoAP] /alert POST failed: {exc}")

def _post_to_phone(alert: dict) -> None:
    """Thread-safe, fire-and-forget CoAP POST."""
    if _coap_loop is None:
        print("[ALERT] CoAP loop not initialised")
        return
    asyncio.run_coroutine_threadsafe(
        _post_coap(json.dumps(alert).encode()),
        _coap_loop,
    )

# =============================================================================
# PUBLIC API
# =============================================================================

def init(coap_loop: asyncio.AbstractEventLoop) -> None:
    """
    Initialise the dispatcher.  Must be called once, after the CoAP server
    event loop is already running.

    coap_loop — the asyncio loop that the CoAP server lives on.
    """
    global _coap_loop
    _coap_loop = coap_loop
    future = asyncio.run_coroutine_threadsafe(_init_client(), _coap_loop)
    future.result(timeout=5.0)
    print("[ALERT] Dispatcher ready (CoAP client + VMS board initialised)")


def send(alert: dict) -> None:
    """
    Deliver one alert to both the phone (CoAP) and the VMS board.

    alert dict keys expected (all produced by main_script.emit_alert):
        type, source, class, object_id, ttrc, ttrc_tier,
        speed_kmh, status, direction, timestamp, alert_message,
        ped_ids_in_frame   (optional list of pedestrian IDs for group detection)
    """
    label            = alert.get("class",            "Unknown")
    object_id        = alert.get("object_id",        -1)
    source           = alert.get("source",           "unknown")
    ped_ids          = alert.get("ped_ids_in_frame", [])

    # ── Phone ─────────────────────────────────────────────────────────────────
    _post_to_phone(alert)

    # ── VMS ───────────────────────────────────────────────────────────────────
    _trigger_vms(label, object_id, source, ped_ids)


def shutdown() -> None:
    """Cancel any pending timers. Call on graceful shutdown."""
    print("[ALERT] Dispatcher shut down")
