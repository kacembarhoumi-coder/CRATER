"""
hazard_map.py — onboard hazard memory: size filter, merging of repeated sightings, priority.

A hazard seen in 30 consecutive frames must become ONE hazard with ONE persistent ID, not 30 packets.
Sightings are merged by inverse-variance weighting (the more certain measurement counts more), so the
position gets BETTER the closer the rover gets. Memory is bounded (max_tracks) as flight software requires.
Dependencies: standard library + crater.telemetry.
"""
import math
from dataclasses import dataclass

from crater.telemetry import Hazard

HAZARD_DEFAULTS = {"rock_min_height_m": 0.1, "rock_min_width_m": 0.2, "crater_min_width_m": 0.3,
                   "show_small": True, "rock_rule": "any"}


def is_hazard(cls, width, height, hf):
    """Same rule as live_localize.py: True if the metric size threatens the rover."""
    if width is None:
        return True
    if cls == "crater":
        return width >= hf["crater_min_width_m"]
    if cls == "rock":
        wide = width >= hf["rock_min_width_m"]
        if height is None:
            return wide
        tall = height >= hf["rock_min_height_m"]
        return (wide and tall) if hf["rock_rule"] == "all" else (wide or tall)
    return True


@dataclass
class Track:
    hid: int
    cls: str
    E: float
    N: float
    unc95: float
    conf: float
    width: float
    hits: int
    first_seen: float
    last_seen: float


class HazardMap:
    def __init__(self, gate_min=0.5, gate_max=3.0, confirm_hits=5, max_tracks=256, reach_m=12.0):
        self.gate_min, self.gate_max = gate_min, gate_max
        self.confirm_hits, self.max_tracks, self.reach_m = confirm_hits, max_tracks, reach_m
        self.tracks = {}
        self._next = 0

    def _new_id(self):
        for _ in range(4096):
            self._next = (self._next + 1) % 4096
            if self._next not in self.tracks:
                return self._next
        raise RuntimeError("hazard ID space exhausted")

    def update(self, dets, t):
        """dets: dicts with cls, E, N (site frame, m), unc95, conf, width. Merges or creates tracks."""
        for d in dets:
            best, best_d = None, 1e9
            for tr in self.tracks.values():
                if tr.cls != d["cls"]:
                    continue
                dist = math.hypot(tr.E - d["E"], tr.N - d["N"])
                gate = min(self.gate_max, max(self.gate_min, d["unc95"] + 0.5 * tr.unc95))   # new measurement's noise sets the gate
                if dist <= gate and dist < best_d:
                    best, best_d = tr, dist
            if best is not None:                                  # fuse: inverse-variance weighting
                w_old, w_new = 1 / max(best.unc95, 0.05) ** 2, 1 / max(d["unc95"], 0.05) ** 2
                best.E = (best.E * w_old + d["E"] * w_new) / (w_old + w_new)
                best.N = (best.N * w_old + d["N"] * w_new) / (w_old + w_new)
                best.unc95 = 1 / math.sqrt(w_old + w_new)
                best.conf = 0.7 * best.conf + 0.3 * d["conf"]
                best.width = 0.7 * best.width + 0.3 * d["width"]
                best.hits += 1
                best.last_seen = t
            else:
                if len(self.tracks) >= self.max_tracks:           # bounded memory: evict the weakest
                    weakest = min(self.tracks.values(), key=lambda x: (x.hits, x.last_seen))
                    del self.tracks[weakest.hid]
                hid = self._new_id()
                self.tracks[hid] = Track(hid, d["cls"], d["E"], d["N"], d["unc95"], d["conf"],
                                         d["width"], 1, t, t)

    def severity(self, tr, rE, rN):
        """0-1 priority: bigger and closer = more critical."""
        size = min(tr.width / 1.0, 1.0)
        near = 1 - min(math.hypot(tr.E - rE, tr.N - rN) / self.reach_m, 1.0)
        return 0.6 * size + 0.4 * near

    def packet_hazards(self, rE, rN, t, limit):
        """Confirmed hazards the packet can encode (within +-reach of the rover), most critical first."""
        cands = [tr for tr in self.tracks.values()
                 if tr.hits >= self.confirm_hits
                 and abs(tr.E - rE) <= self.reach_m and abs(tr.N - rN) <= self.reach_m]
        cands.sort(key=lambda tr: self.severity(tr, rE, rN), reverse=True)
        return [Hazard(tr.hid, tr.cls, tr.E, tr.N, tr.unc95, tr.conf, tr.width, age=max(0.0, t - tr.last_seen))
                for tr in cands[:limit]]
