"""
coap_server.py  —  InterSafe-5G  CoAP Server Module
=====================================================
Owns the CoAP *server* side only:
    /alert      (AlertResource)     — phone Observe target; alert_dispatcher
                                       POSTs each alert here as a CoAP client
    /gps        (GPSResource)       — phone GPS updates
    /heartbeat  (HeartbeatResource) — connection check

This was pulled out of main_script.py so that main_script only owns
sensor polling + TTRC logic, as its own docstring claims. The CoAP
*client* (posting alerts into this server) and the VMS board still
live in alert_dispatcher.py — this module is server-resources only.

Usage (from main_script.py):
    import coap_server
    loop = coap_server.start()          # starts server thread, returns its loop
    alert_dispatcher.init(loop)         # dispatcher's CoAP client attaches to it
"""

import asyncio
import json
import math
import threading
import time

import aiocoap
import aiocoap.resource as resource
from aiocoap import Context, Message
from aiocoap.numbers.codes import Code

# =============================================================================
# CONFIG
# =============================================================================

COAP_BIND_IP     = "0.0.0.0"
COAP_PORT        = 5683
GPS_MIN_INTERVAL = 5          # seconds between accepted GPS updates

# =============================================================================
# GPS STATE  (module-level; read via coap_server.latest_phone_gps if needed)
# =============================================================================

latest_phone_gps = {}
_gps_last_time   = None
_gps_prev_lat    = _gps_prev_lon = _gps_prev_time = None


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _safe_decode(payload: bytes, route: str) -> str | None:
    """
    Decode a CoAP payload as UTF-8, returning None (and logging) on failure
    instead of letting UnicodeDecodeError crash the render_post coroutine.
    """
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"⚠️  {route} payload is not valid UTF-8 — ignored ({exc})")
        return None


# =============================================================================
# CoAP SERVER RESOURCES
# =============================================================================

class AlertResource(resource.ObservableResource):
    """Phone registers here via CoAP Observe; dispatcher POSTs here per alert."""

    def __init__(self):
        super().__init__()
        self.alert_message   = b""
        self._last_forwarded = b""

    async def render_get(self, request):
        print("📱 Mobile registered for Observe (/alert)")
        resp             = Message(payload=self.alert_message, code=Code.CONTENT)
        resp.opt.observe = 0
        return resp

    async def render_post(self, request):
        raw_text = _safe_decode(request.payload, "/alert")
        if raw_text is None:
            return Message(code=Code.BAD_REQUEST, payload=b"Invalid Encoding")

        new_message = raw_text.strip()
        print(f"\n📩 Alert received → /alert : {new_message}")

        try:
            json.loads(new_message)
        except json.JSONDecodeError:
            print("⚠️  Non-JSON payload — ignored.")
            return Message(code=Code.BAD_REQUEST, payload=b"Invalid JSON")

        encoded = new_message.encode()
        if encoded == self._last_forwarded:
            print("⚠️  Duplicate payload — not re-notifying observers.")
            return Message(code=Code.CHANGED, payload=b"Duplicate Ignored")

        self.alert_message   = encoded
        self._last_forwarded = encoded
        print("🚀 Notifying observers...")
        self.updated_state()
        return Message(code=Code.CHANGED, payload=b"Alert Updated")


class GPSResource(resource.Resource):

    async def render_post(self, request):
        global latest_phone_gps, _gps_last_time
        global _gps_prev_lat, _gps_prev_lon, _gps_prev_time

        raw_text = _safe_decode(request.payload, "/gps")
        if raw_text is None:
            return Message(code=Code.BAD_REQUEST, payload=b"Invalid Encoding")

        try:
            data        = json.loads(raw_text)
            lat         = data.get("lat")
            lon         = data.get("lon")
            phone_speed = data.get("speed")
            direction   = data.get("direction") or "Stationary"
            accuracy    = data.get("accuracy")

            try:
                accuracy = float(accuracy)
            except (TypeError, ValueError):
                accuracy = None

            if lat is None or lon is None:
                raise ValueError("lat/lon missing")

            if accuracy is not None and accuracy > 500:
                print("⚠️  GPS ignored (accuracy > 500 m)")
                return Message(code=Code.CHANGED, payload=b"Low Accuracy Ignored")

            now = time.time()
            if _gps_last_time is not None:
                dt = now - _gps_last_time
                if dt < GPS_MIN_INTERVAL:
                    return Message(code=Code.CHANGED, payload=b"Rate Limited")
                freq = 1 / dt
            else:
                dt = freq = None

            _gps_last_time = now

            server_speed = None
            if _gps_prev_lat is not None:
                dist = _haversine(_gps_prev_lat, _gps_prev_lon, lat, lon)
                elapsed = now - _gps_prev_time
                if elapsed > 0:
                    server_speed = (dist / elapsed) * 3.6

            _gps_prev_lat  = lat
            _gps_prev_lon  = lon
            _gps_prev_time = now

            latest_phone_gps = {
                "lat": lat, "lon": lon,
                "phone_speed": phone_speed, "server_speed": server_speed,
                "direction": direction, "accuracy": accuracy,
            }

            print("\n📍 GPS Data Received")
            print(f"   Latitude  : {lat}")
            print(f"   Longitude : {lon}")
            print(f"   Speed     : {round(phone_speed, 2) if phone_speed else 0} km/h (phone)"
                  f"{f'  |  {server_speed:.2f} km/h (server)' if server_speed else ''}")
            print(f"   Direction : {direction}")
            print(f"   Accuracy  : {round(accuracy, 2) if accuracy else 'Unknown'} m")
            if freq is not None:
                print(f"   Interval  : {dt:.2f}s  ({freq:.2f} Hz)")
            print("--------------------------------------------------")

            return Message(code=Code.CHANGED, payload=b"GPS Received")

        except json.JSONDecodeError:
            print("❌ Invalid JSON from GPS")
            return Message(code=Code.BAD_REQUEST, payload=b"Invalid JSON")
        except Exception as exc:
            print("❌ GPS error:", exc)
            return Message(code=Code.BAD_REQUEST, payload=b"Processing Error")


class HeartbeatResource(resource.Resource):
    async def render_get(self, request):
        print("💓 Heartbeat from mobile")
        return Message(code=Code.CONTENT, payload=b"alive")


# =============================================================================
# CoAP SERVER  (runs in its own thread + event loop)
# =============================================================================

_coap_server_loop = asyncio.new_event_loop()
_alert_resource: AlertResource | None = None


def _run_coap_server():
    asyncio.set_event_loop(_coap_server_loop)
    _coap_server_loop.run_until_complete(_start_coap_server())
    _coap_server_loop.run_forever()


async def _start_coap_server():
    global _alert_resource
    _alert_resource  = AlertResource()
    root = resource.Site()
    root.add_resource(["alert"],     _alert_resource)
    root.add_resource(["gps"],       GPSResource())
    root.add_resource(["heartbeat"], HeartbeatResource())
    await Context.create_server_context(root, bind=(COAP_BIND_IP, COAP_PORT))
    print("🚀 CoAP Server Running")
    print(f"   Listening on UDP {COAP_PORT}")
    print("   /alert     → Observe (conflict alerts to UE)")
    print("   /gps       → POST    (GPS from UE)")
    print("   /heartbeat → GET     (connection check)")
    print("--------------------------------------------------")


# =============================================================================
# PUBLIC API
# =============================================================================

def start() -> asyncio.AbstractEventLoop:
    """
    Start the CoAP server thread + event loop.

    Returns the loop so the caller (main_script.py) can hand it to
    alert_dispatcher.init(loop) for the CoAP client side.
    """
    threading.Thread(
        target=_run_coap_server, daemon=True, name="coap-server"
    ).start()
    time.sleep(1.0)      # allow server to bind before client context is created
    return _coap_server_loop
