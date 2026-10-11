"""
link.py — lossy radio link emulator (team-implemented UDP relay).

rover --UDP--> [ this relay ] --UDP--> ground station
Emulates: random packet loss, latency + jitter, bandwidth cap (serialization delay + bounded queue),
blackout windows (everything dropped), optional bit corruption (exercises the CRC).
Parameters can be changed live (set_params / blackout), e.g. from the dashboard.
Dependencies: standard library only.
"""
import heapq
import random
import socket
import threading
import time

LINK_DEFAULTS = {"loss": 0.15, "delay_ms": 400, "jitter_ms": 150, "bandwidth_bps": 2400,
                 "queue_max": 32, "corrupt": 0.0}


class LinkEmulator(threading.Thread):
    def __init__(self, listen, dest, seed=1, **params):
        super().__init__(daemon=True)
        self.p = {**LINK_DEFAULTS, **params}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(listen)
        self.sock.settimeout(0.01)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = dest
        self.rng = random.Random(seed)                      # seeded: reproducible tests
        self.heap, self.n, self.busy_until, self.blackout_until = [], 0, 0.0, 0.0
        self.lock = threading.Lock()
        self.stats = {"sent": 0, "delivered": 0, "dropped_loss": 0, "dropped_blackout": 0,
                      "dropped_queue": 0, "corrupted": 0, "bytes_in": 0, "bytes_out": 0,
                      "delay_sum": 0.0, "last_delay": 0.0}
        self._halt = threading.Event()

    def set_params(self, **kw):
        with self.lock:
            self.p.update({k: v for k, v in kw.items() if k in LINK_DEFAULTS})

    def blackout(self, seconds):
        with self.lock:
            self.blackout_until = time.monotonic() + seconds

    def in_blackout(self):
        return time.monotonic() < self.blackout_until

    def snapshot(self):
        with self.lock:
            return {**self.stats, "params": dict(self.p), "blackout": self.in_blackout()}

    def stop(self):
        self._halt.set()

    def _ingest(self, data, now):
        with self.lock:
            p, st = dict(self.p), self.stats
            st["sent"] += 1
            st["bytes_in"] += len(data)
            if now < self.blackout_until:
                st["dropped_blackout"] += 1
                return
            if self.rng.random() < p["loss"]:
                st["dropped_loss"] += 1
                return
            if len(self.heap) >= p["queue_max"]:
                st["dropped_queue"] += 1
                return
            if self.rng.random() < p["corrupt"]:
                i = self.rng.randrange(len(data))
                data = data[:i] + bytes([data[i] ^ (1 << self.rng.randrange(8))]) + data[i + 1:]
                st["corrupted"] += 1
            tx_time = len(data) * 8 / p["bandwidth_bps"]              # serialization delay
            self.busy_until = max(now, self.busy_until) + tx_time
            deliver = self.busy_until + max(0.0, p["delay_ms"] + self.rng.uniform(-p["jitter_ms"], p["jitter_ms"])) / 1000
            self.n += 1
            heapq.heappush(self.heap, (deliver, self.n, data, now))

    def run(self):
        while not self._halt.is_set():
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                data = None
            except OSError:
                break
            now = time.monotonic()
            if data is not None:
                self._ingest(data, now)
            while True:
                with self.lock:
                    if not self.heap or self.heap[0][0] > time.monotonic():
                        break
                    _, _, pkt, t_in = heapq.heappop(self.heap)
                    d = time.monotonic() - t_in
                    self.stats["delivered"] += 1
                    self.stats["bytes_out"] += len(pkt)
                    self.stats["delay_sum"] += d
                    self.stats["last_delay"] = d
                self.out.sendto(pkt, self.dest)
