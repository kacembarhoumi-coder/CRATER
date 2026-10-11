"""
ground.py — Mission Control ground station: UDP receiver + web dashboard server.

Receives packets, verifies CRC (corrupted packets are counted and discarded), tracks sequence gaps to
estimate the real packet loss, keeps the last known state of every hazard and computes DATA AGE
(age at transmission + time since reception). Serves web/dashboard.html and /api/state (JSON).
Dependencies: standard library + crater.telemetry.
"""
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from crater.telemetry import PacketError, decode_packet

WEB = Path(__file__).resolve().parent.parent / "web"


class VideoFeed:
    """Latest annotated rover camera frame (JPEG bytes), shared between the rover loop and the web server."""

    def __init__(self):
        self.lock, self.jpeg, self.n = threading.Lock(), None, 0

    def set(self, jpeg):
        with self.lock:
            self.jpeg, self.n = jpeg, self.n + 1

    def get(self):
        with self.lock:
            return self.jpeg, self.n


class GroundStation(threading.Thread):
    def __init__(self, port, link, max_hazards=512):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.link, self.max_hazards = link, max_hazards
        self.lock = threading.Lock()
        self.hazards, self.rover, self.trail = {}, None, deque(maxlen=800)
        self.window = deque(maxlen=60)                          # 1 = received, 0 = lost (from seq gaps)
        self.last_seq, self.last_rx, self.rx, self.bad, self.bytes_rx, self.dups = None, None, 0, 0, 0, 0
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def run(self):
        while not self._halt.is_set():
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self.ingest(data, time.time())

    def ingest(self, data, now):
        """Process one received datagram. Called by the receiver thread, or directly by virtual-time tests."""
        try:
            pkt = decode_packet(data)
        except PacketError:
            with self.lock:
                self.bad += 1
            return
        with self.lock:
            if self.last_seq is not None:
                delta = (pkt["seq"] - self.last_seq) & 0xFFFF
                if delta == 0:
                    self.dups += 1
                    return
                if delta >= 0x8000:                             # late (reordered) packet: ignore for loss
                    return
                self.window.extend([0] * min(delta - 1, 60))
            self.window.append(1)
            self.last_seq, self.last_rx = pkt["seq"], now
            self.rx += 1
            self.bytes_rx += len(data)
            self.rover = {**pkt["rover"], "t": pkt["t"], "seq": pkt["seq"]}
            self.trail.append((pkt["rover"]["E"], pkt["rover"]["N"]))
            for h in pkt["hazards"]:
                self.hazards[h["hid"]] = {**h, "rx": now}
            if len(self.hazards) > self.max_hazards:
                del self.hazards[min(self.hazards, key=lambda k: self.hazards[k]["rx"])]

    def snapshot(self):
        now, lk = time.time(), self.link.snapshot()
        with self.lock:
            since = None if self.last_rx is None else now - self.last_rx
            status = "NO DATA" if since is None else "LOST" if since > 8 else "DEGRADED" if since > 3 else "OK"
            loss = 100 * (1 - sum(self.window) / len(self.window)) if self.window else 0.0
            hz = [{"hid": h["hid"], "cls": h["cls"], "E": h["E"], "N": h["N"], "unc95": h["unc95"],
                   "conf": round(h["conf"], 2), "width": h["width"], "age": round(h["age"] + now - h["rx"], 1)}
                  for h in self.hazards.values()]
            return {"rover": self.rover, "trail": list(self.trail), "hazards": hz,
                    "link": {"status": status, "since_last_s": since, "loss_pct": round(loss, 1),
                             "latency_ms": round(lk["last_delay"] * 1000), "rx": self.rx, "corrupted": self.bad,
                             "bytes_rx": self.bytes_rx, "blackout": lk["blackout"],
                             "dropped": lk["dropped_loss"] + lk["dropped_blackout"] + lk["dropped_queue"],
                             "set_loss_pct": round(lk["params"]["loss"] * 100)}}


def serve(station, link, port, feed=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._send(json.dumps(station.snapshot()).encode(), "application/json")
            elif self.path.startswith("/video.mjpg") and feed is not None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                last = -1
                try:
                    while True:
                        jpeg, n = feed.get()
                        if jpeg is None or n == last:
                            time.sleep(0.03)
                            continue
                        last = n
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except OSError:                                 # browser closed the tab
                    return
            elif self.path.startswith("/api/frame.jpg") and feed is not None:
                jpeg, _ = feed.get()
                if jpeg is not None:
                    self._send(jpeg, "image/jpeg")
                else:
                    self.send_error(503, "No frame available yet")
            elif self.path in ("/", "/index.html"):
                self._send((WEB / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/link":
                return self.send_error(404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if "loss" in body:
                    link.set_params(loss=min(max(float(body["loss"]), 0.0), 0.95))
                if "blackout" in body:
                    link.blackout(min(max(float(body["blackout"]), 0.0), 120.0))
            except (ValueError, TypeError):
                return self.send_error(400)
            self._send(b"{}", "application/json")

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
