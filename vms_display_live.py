"""
vms_display_live.py
===============
UFD Board Class — Shared Library
Imported by both vms_send.py and vms_receive.py (both run on MEC server).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NETWORK TOPOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MEC Server (cloud)
  ├── vms_send.py      ──TCP 127.0.0.1:9000──►
  └── vms_receive.py  ──TCP 10.45.0.53:5000──►  (over 5G)
                                                      │
                                                 gateway (gateway (CPE/RUT or similar) Router
                                                 Cellular IP : 10.45.0.53  ← MEC connects here
                                                 LAN IP      : 192.168.1.1
                                                 Port-forward: TCP:5000 → 192.168.1.75:5000
                                                      │ (LAN / Ethernet)
                                                 VMS Board
                                                 LAN IP      : 192.168.1.75
                                                 UFD TCP port: 5000

WHY WE TARGET 10.45.0.53 AND NOT 192.168.1.75:
    192.168.1.75 is a PRIVATE LAN IP — it is unreachable from the MEC
    server across the internet / 5G core. The only address the MEC can
    reach is the gateway (gateway (CPE/RUT or similar's cellular (WAN) IP: 10.45.0.53.
    The gateway (CPE/RUT or similar must have a port-forward rule so that any TCP connection
    arriving on 10.45.0.53:5000 is forwarded to 192.168.1.75:5000.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATEWAY PORT-FORWARD RULE  (configure once on the gateway (CPE/RUT or similar /router UI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Protocol  : TCP
    WAN port  : 5000          (MEC connects to 10.45.0.53:5000)
    LAN IP    : 192.168.1.75  (VMS board)
    LAN port  : 5000          (UFD board TCP port)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UFD PROTOCOL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Send 20-byte binary header (5 x little-endian uint32):
    header : 0x5566AABB
    size   : len(command string)
    cmd    : 0x0000000C  (fixed)
    info   : 0x00000000  (fixed)
    crc    : size XOR cmd XOR info

STEP 2 — Receive "#OK*" from board (4 bytes).
STEP 3 — Send ASCII command string.
STEP 4 — Receive "#OK*" (success) or "#ER*" (error).

Command string format:
    |C|4|<mode>|<line1>[~<line2>]|<speed>|<font>|[<image>|]
    mode  : 1=static  2=scrolling  9=blinking
    speed : 1-7
    font  : 1=Regular  2=Large  3=Full
    image : ok  or  er  (optional)

Other commands:
    |C|6|      → clear static text
    |C|7|      → clear scrolling text
    |C|8|ok|   → show green OK image
    |C|8|er|   → show red ERR image
    |C|9|      → clear all
    |C|R|      → reset display
"""

import socket
import struct
import time
import threading
import argparse

# ── UFD Protocol constants ────────────────────────────────────────────────────
HEADER        = 0x5566AABB
CMD_FIXED     = 0x0000000C
INFO_FIXED    = 0x00000000
ACK_OK        = b"#OK*"
ACK_ER        = b"#ER*"
PACKET_FORMAT = "<IIIII"      # 5 x uint32 little-endian = 20 bytes

# ── IP / Port constants ───────────────────────────────────────────────────────
GATEWAY_WAN_IP  = "10.45.0.56"    # gateway (CPE/RUT or similar WAN/cellular IP — reachable from MEC over 5G
GATEWAY_LAN_IP  = "192.168.1.1"   # gateway (CPE/RUT or similar LAN gateway          (reference only)
VMS_BOARD_LAN_IP = "192.168.1.75"  # VMS board LAN IP         (NOT reachable from MEC)
UFD_PORT         = 5000            # UFD TCP port on VMS board (same on gateway (CPE/RUT or similar forward)

# !! IMPORTANT: All UFD TCP connections from the MEC server go to GATEWAY_WAN_IP:UFD_PORT
# !! The gateway (CPE/RUT or similar port-forwards them to VMS_BOARD_LAN_IP:UFD_PORT over its local LAN.
UFD_TARGET_IP   = GATEWAY_WAN_IP
UFD_TARGET_PORT = UFD_PORT

# ── Blink defaults ────────────────────────────────────────────────────────────
DEFAULT_BLINK_ON_SEC  = 1.0   # seconds display is ON  (red alert) per cycle
DEFAULT_BLINK_OFF_SEC = 0.5   # seconds display is OFF (blank)     per cycle

# ── Internal thread guard ─────────────────────────────────────────────────────
# Max seconds to wait for blink thread exit. Safety guard — not blink duration.
THREAD_JOIN_TIMEOUT = 3


# ── Low-level UFD helpers ─────────────────────────────────────────────────────

def build_cmd_packet(data: bytes) -> bytes:
    """Build the 20-byte binary header packet required by the UFD protocol."""
    size = len(data)
    crc  = size ^ CMD_FIXED ^ INFO_FIXED
    return struct.pack(PACKET_FORMAT, HEADER, size, CMD_FIXED, INFO_FIXED, crc)


def send_command(ip: str, port: int, command_str: str, timeout: float = 5.0) -> bool:
    """
    Open a fresh TCP connection, send one UFD command, and close.

    ip   : MUST be GATEWAY_WAN_IP (10.45.0.53) when called from MEC.
           192.168.1.75 is unreachable from MEC — never use it here.
    port : UFD_PORT (5000).  gateway (CPE/RUT or similar forwards this port to the VMS board.

    Returns True on success, False on any error.
    """
    data = command_str.encode("utf-8")
    pkt  = build_cmd_packet(data)

    try:
        with socket.create_connection((ip, port), timeout=timeout) as sock:

            # Step 1: send 20-byte header
            sock.sendall(pkt)

            # Step 2: wait for #OK*
            ack = sock.recv(4)
            if ack != ACK_OK:
                print(f"[UFD][ERROR] Expected #OK* after header, got: {ack!r}")
                return False

            time.sleep(0.1)   # small pause — board needs a moment between steps

            # Step 3: send ASCII command string
            sock.sendall(data)

            # Step 4: wait for #OK* or #ER*
            ack2 = sock.recv(4)
            if ack2 == ACK_OK:
                print(f"[UFD][OK]   {command_str!r}")
                return True
            elif ack2 == ACK_ER:
                print(f"[UFD][WARN] Board returned #ER* for: {command_str!r}")
                return False
            else:
                print(f"[UFD][ERROR] Unexpected second ack: {ack2!r}")
                return False

    except socket.timeout:
        print(f"[UFD][ERROR] Timed out connecting to {ip}:{port}")
        print(f"             Check: (1) gateway (CPE/RUT or similar cellular IP is correct "
              f"(2) gateway (CPE/RUT or similar port-forward TCP:{port} → {VMS_BOARD_LAN_IP}:{port} is set")
        return False
    except ConnectionRefusedError:
        print(f"[UFD][ERROR] Connection REFUSED at {ip}:{port}")
        print(f"             gateway (CPE/RUT or similar is reachable but nothing is listening / "
              f"forwarding TCP:{port}. Check GATEWAY PORT-FORWARD RULE.")
        return False
    except OSError as exc:
        print(f"[UFD][ERROR] Network OS error to {ip}:{port} — {exc}")
        return False
    except Exception as exc:
        print(f"[UFD][ERROR] Unexpected: {exc}")
        return False


# ── UFDBoard class ────────────────────────────────────────────────────────────

class UFDBoard:
    """
    High-level interface to the Envoys UFD / VMS board.

    On MEC + 5G + gateway (CPE/RUT or similar deployments, always construct with the gateway (CPE/RUT or similar's
    cellular IP and UFD port:

        board = UFDBoard()                           # uses built-in defaults
        board = UFDBoard("10.45.0.53", 5000)        # explicit

    The gateway (CPE/RUT or similar at 10.45.0.53 port-forwards TCP:5000 → 192.168.1.75:5000,
    so the VMS board responds as if the MEC connected directly.
    """

    def __init__(self, ip: str = UFD_TARGET_IP, port: int = UFD_TARGET_PORT,
                 timeout: float = 5.0):
        self.ip      = ip
        self.port    = port
        self.timeout = timeout

        self._stop_event   = threading.Event()
        self._blink_thread = None
        self._blink_lock   = threading.Lock()

        print(f"[UFD] Board target : {self.ip}:{self.port}  "
              f"(gateway (CPE/RUT or similar cellular IP → port-forward → VMS board {VMS_BOARD_LAN_IP}:{self.port})")

    def _send(self, cmd: str) -> bool:
        return send_command(self.ip, self.port, cmd, self.timeout)

    # ── Display commands ──────────────────────────────────────────────────────

    def display_message(self, line1: str, line2: str = "",
                        scroll: bool = False, blink: bool = False,
                        speed: int = 1, font: int = 1, image: str = "") -> bool:
        """Send a text message to the VMS board."""
        text = line1 + (f"~{line2}" if line2 else "")
        mode = 9 if blink else (2 if scroll else 1)
        cmd  = (f"|C|4|{mode}|{text}|{speed}|{font}|{image}|" if image
                else f"|C|4|{mode}|{text}|{speed}|{font}|")
        return self._send(cmd)

    def clear_static(self)     -> bool: return self._send("|C|6|")
    def clear_scrolling(self)  -> bool: return self._send("|C|7|")
    def clear_all(self)        -> bool: return self._send("|C|9|")
    def reset_display(self)    -> bool: return self._send("|C|R|")
    def show_ok_image(self)    -> bool: return self._send("|C|8|ok|")
    def show_error_image(self) -> bool: return self._send("|C|8|er|")
    def send_raw(self, cmd: str) -> bool: return self._send(cmd)

    # ── Blink management ─────────────────────────────────────────────────────

    def _blink_worker(self, line1: str, line2: str,
                      blink_on: float, blink_off: float):
        """
        Background thread: cycles ON (red alert + text) / OFF (blank)
        until _stop_event is set.
        """
        self._stop_event.clear()
        print(f"[UFD] Blink started → '{line1} {line2}'  "
              f"ON={blink_on}s  OFF={blink_off}s")

        while not self._stop_event.is_set():
            self.display_message(line1, line2, image="er")   # ON  — red + text
            if self._stop_event.wait(timeout=blink_on):
                break
            self.clear_all()                                  # OFF — blank
            if self._stop_event.wait(timeout=blink_off):
                break

        self.clear_all()
        print(f"[UFD] Blink stopped.")

    def start_blink(self, line1: str, line2: str = "",
                    blink_on: float  = DEFAULT_BLINK_ON_SEC,
                    blink_off: float = DEFAULT_BLINK_OFF_SEC):
        """
        Start a blinking alert in a background thread.
        If a blink is already running, it is cancelled and replaced.
        """
        with self._blink_lock:
            if self._blink_thread is not None and self._blink_thread.is_alive():
                self._stop_event.set()
                self._blink_thread.join(timeout=THREAD_JOIN_TIMEOUT)

            self._stop_event.clear()
            self._blink_thread = threading.Thread(
                target=self._blink_worker,
                args=(line1, line2, blink_on, blink_off),
                daemon=True
            )
            self._blink_thread.start()

    def stop_blink(self):
        """Stop blinking and restore the green OK (idle) image."""
        with self._blink_lock:
            if self._blink_thread is not None and self._blink_thread.is_alive():
                self._stop_event.set()
                self._blink_thread.join(timeout=THREAD_JOIN_TIMEOUT)

        time.sleep(0.3)        # let board settle after clear
        self.show_ok_image()
        print(f"[UFD] Board → GREEN OK (idle)")


# ── Interactive CLI ───────────────────────────────────────────────────────────

def interactive_demo(board: UFDBoard):
    print("\n" + "="*60)
    print("  UFD Board Interactive Controller")
    print(f"  Target : {board.ip}:{board.port}")
    print(f"           gateway (CPE/RUT or similar → VMS board {VMS_BOARD_LAN_IP} via LAN port-forward")
    print("="*60)
    while True:
        print("\n  1. Static message        6. Clear all")
        print("  2. Scrolling message     7. Reset display")
        print("  3. Blink message (board) 8. Send raw command")
        print("  4. Show OK image         9. Start blink alert (thread)")
        print("  5. Show ERR image       10. Stop blink alert")
        print("  0. Exit")
        c = input("\nChoice: ").strip()
        if   c == "0": break
        elif c == "1":
            board.display_message(input("  Line1: ").strip(),
                                  input("  Line2: ").strip())
        elif c == "2":
            board.display_message(input("  Line1: ").strip(),
                                  input("  Line2: ").strip(),
                                  scroll=True,
                                  speed=int(input("  Speed 1-7 (default 3): ").strip() or "3"))
        elif c == "3":
            board.display_message(input("  Line1: ").strip(),
                                  input("  Line2: ").strip(), blink=True)
        elif c == "4":  board.show_ok_image()
        elif c == "5":  board.show_error_image()
        elif c == "6":  board.clear_all()
        elif c == "7":  board.reset_display()
        elif c == "8":  board.send_raw(input("  Raw command: ").strip())
        elif c == "9":
            board.start_blink(
                input("  Line1 (e.g. PERSON): ").strip(),
                input("  Line2 (e.g. DETECTED): ").strip(),
                blink_on  = float(input(f"  ON  sec [{DEFAULT_BLINK_ON_SEC}]: ").strip()  or DEFAULT_BLINK_ON_SEC),
                blink_off = float(input(f"  OFF sec [{DEFAULT_BLINK_OFF_SEC}]: ").strip() or DEFAULT_BLINK_OFF_SEC),
            )
        elif c == "10": board.stop_blink()
        else:           print("  Invalid choice.")


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UFD Board direct controller (standalone)")
    parser.add_argument("--ip",          default=UFD_TARGET_IP,   help=f"Target IP (default: {UFD_TARGET_IP})")
    parser.add_argument("--port",        default=UFD_TARGET_PORT, type=int)
    parser.add_argument("--msg",         default="")
    parser.add_argument("--line2",       default="")
    parser.add_argument("--scroll",      action="store_true")
    parser.add_argument("--blink",       action="store_true")
    parser.add_argument("--speed",       default=3, type=int)
    parser.add_argument("--font",        default=1, type=int)
    parser.add_argument("--image",       default="")
    parser.add_argument("--clear",       action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    board = UFDBoard(args.ip, args.port)

    if   args.clear:       board.clear_all()
    elif args.msg:         board.display_message(args.msg, args.line2,
                                                  scroll=args.scroll, blink=args.blink,
                                                  speed=args.speed, font=args.font,
                                                  image=args.image)
    elif args.interactive: interactive_demo(board)
    else:                  print("No action. Use --msg, --clear, or --interactive.")


if __name__ == "__main__":
    main()