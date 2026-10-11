"""
link.py — lossy radio link emulator (team-implemented UDP relay).

rover --UDP--> [ this relay ] --UDP--> ground station
Emulates: random or BURST packet loss, latency + jitter, bandwidth cap (serialization delay + bounded
queue), blackout windows (everything dropped), optional bit corruption (exercises the CRC).
Parameters can be changed live (set_params / blackout), e.g. from the dashboard.
Dependencies: standard library only.
"""
import heapq
import random
import socket
import threading
import time

LINK_DEFAULTS = {"loss": 0.15, "delay_ms": 400, "jitter_ms": 150, "bandwidth_bps": 2400,
                 "queue_max": 32, "corrupt": 0.0, "burst": 1.0}   # burst = mean burst length (1 = independent loss)


class LinkModel:
    """Virtual-time link simulator (deterministic, no sockets, runs in simulated time)."""

    def __init__(self, seed=1, **params):
        self.p = {**LINK_DEFAULTS, **params}
        self.rng = random.Random(seed)
        self.busy_until = 0.0
        self.blackout_until = 0.0
        self.bad = False
        self.queue_count = 0
        self.stats = {"sent": 0, "delivered": 0, "dropped_loss": 0, "dropped_blackout": 0,
                      "dropped_queue": 0, "corrupted": 0, "bytes_in": 0, "bytes_out": 0,
                      "delay_sum": 0.0, "last_delay": 0.0}

    def _lost(self, p):
        burst = p.get("burst", 1.0)
        if burst > 1.0:
            p_bg = 1.0 / burst
            p_gb = min(1.0, p["loss"] * p_bg / max(1e-9, 1 - p["loss"]))
            if self.bad:
                self.bad = self.rng.random() >= p_bg
            else:
                self.bad = self.rng.random() < p_gb
            return self.bad
        return self.rng.random() < p["loss"]

    def process(self, data, now):
        """Processes one outgoing packet at virtual time `now`.
        Returns (data, deliver_time) or None if dropped."""
        p, st = self.p, self.stats
        st["sent"] += 1
        st["bytes_in"] += len(data)
        if now < self.blackout_until:
            st["dropped_blackout"] += 1
            return None
        if self._lost(p):
            st["dropped_loss"] += 1
            return None
        if self.queue_count >= p["queue_max"]:
            st["dropped_queue"] += 1
            return None
        if self.rng.random() < p["corrupt"]:
            i = self.rng.randrange(len(data))
            data = data[:i] + bytes([data[i] ^ (1 << self.rng.randrange(8))]) + data[i + 1:]
            st["corrupted"] += 1
        tx_time = len(data) * 8 / p["bandwidth_bps"]
        self.busy_until = max(now, self.busy_until) + tx_time
        deliver = self.busy_until + max(0.0, p["delay_ms"] + self.rng.uniform(-p["jitter_ms"], p["jitter_ms"])) / 1000
        self.queue_count += 1
        return data, deliver

    def delivered(self, n_bytes, delay):
        """Record packet departure from the queue and arrival at the receiver."""
        self.queue_count = max(0, self.queue_count - 1)
        self.stats["delivered"] += 1
        self.stats["bytes_out"] += n_bytes
        self.stats["delay_sum"] += delay
        self.stats["last_delay"] = delay


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
        self.bad = False                                    # burst-loss state (True = in a loss burst)
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

    def _lost(self, p):
        """Loss decision. burst > 1: two-state Gilbert-Elliott model (mean burst length = burst,
        long-run loss rate = loss). Otherwise independent random loss."""
        burst = p.get("burst", 1.0)
        if burst > 1.0:
            p_bg = 1.0 / burst                                           # bad -> good probability
            p_gb = min(1.0, p["loss"] * p_bg / max(1e-9, 1 - p["loss"]))  # good -> bad probability
            if self.bad:
                self.bad = self.rng.random() >= p_bg
            else:
                self.bad = self.rng.random() < p_gb
            return self.bad
        return self.rng.random() < p["loss"]

    def _ingest(self, data, now):
        with self.lock:
            p, st = dict(self.p), self.stats
            st["sent"] += 1
            st["bytes_in"] += len(data)
            if now < self.blackout_until:
                st["dropped_blackout"] += 1
                return
            if self._lost(p):
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
                try:
                    self._ingest(data, now)
                except Exception as e:                      # never let one bad packet kill the relay
                    print("link error:", e)
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