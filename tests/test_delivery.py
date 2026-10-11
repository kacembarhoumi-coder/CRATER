"""
test_delivery.py — how many hazards survive the lossy link? (virtual time: a 5-minute mission in ~1 s)

Replays a scripted mission through the REAL code path: HazardMap -> encode_packet -> LinkModel (loss, bursts,
delay, bandwidth, blackout, corruption) -> GroundStation.ingest (CRC, sequence gaps, hazard memory).
Same link settings as the SRT benchmark (1.3 s one-way delay, 9600 bit/s, bursts of 3 packets, seed 1).

    pytest tests/test_delivery.py -q        # assertions
    python tests/test_delivery.py           # writes test_outputs/delivery_report.json and prints the table
Reports the BASELINE scheme: every packet is self-contained and re-sent every second (repetition only,
no ACK, no retransmission).
"""
import heapq
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from crater.ground import GroundStation  # noqa: E402
from crater.hazard_map import HazardMap  # noqa: E402
from crater.link import LinkModel  # noqa: E402
from crater.telemetry import RoverState, encode_packet, max_hazards_for  # noqa: E402

SPEED, DT, TX_PERIOD, DURATION, DRAIN = 0.1, 0.1, 1.0, 300.0, 20.0
LINK = {"delay_ms": 1300, "jitter_ms": 150, "bandwidth_bps": 9600, "queue_max": 32, "burst": 3.0}


def make_truth(seed, n=20, length=30.0):
    rng = random.Random(seed)
    xs = sorted(rng.uniform(3, length) for _ in range(n))
    return [{"cls": rng.choice(["rock", "rock", "crater"]), "E": x, "N": rng.uniform(-2, 2),
             "width": rng.uniform(0.3, 0.8)} for x in xs]


def run_scenario(loss, seed=1, corrupt=0.0, blackout=(120.0, 135.0), max_packet_bytes=80):
    rng = random.Random(seed * 7 + 1)
    truth = make_truth(seed)
    hmap, max_h = HazardMap(confirm_hits=5), max_hazards_for(max_packet_bytes)
    link = LinkModel(seed=seed, loss=loss, corrupt=corrupt, **LINK)
    ground = GroundStation(0, None)                           # ephemeral port, never started: ingest() only
    inflight, n, seq, next_tx, t = [], 0, 0, 0.0, 0.0
    confirmed_at, first_rx, pkts, hz_sent, blackout_set = {}, {}, 0, 0, False

    while t < DURATION + DRAIN:
        rE = SPEED * min(t, DURATION)
        if blackout and not blackout_set and t >= blackout[0]:
            link.blackout_until, blackout_set = blackout[1], True
        if t < DURATION:                                       # simulated detector: noisy sightings ahead
            obs = []
            for h in truth:
                d = h["E"] - rE
                if 1.0 <= d <= 8.0 and rng.random() < 0.8:
                    sig = 0.022 * d * d + 0.05                 # range error ~ distance squared
                    obs.append({"cls": h["cls"], "E": h["E"] + rng.gauss(0, sig),
                                "N": h["N"] + rng.gauss(0, 0.2 * sig + 0.05), "unc95": 2 * sig,
                                "conf": 0.8, "width": h["width"]})
            hmap.update(obs, t)
            for tr in hmap.tracks.values():
                if tr.hits >= hmap.confirm_hits and tr.hid not in confirmed_at:
                    confirmed_at[tr.hid] = t
        if t >= next_tx:                                       # TRANSMIT
            hz = hmap.packet_hazards(rE, 0.0, t, max_h)
            pkt = encode_packet(seq, 1, RoverState(t=t, E=rE, N=0.0, yaw_deg=0.0, pose_sigma=0.05), hz)
            seq, next_tx, pkts, hz_sent = seq + 1, next_tx + TX_PERIOD, pkts + 1, hz_sent + len(hz)
            res = link.process(pkt, t)
            if res is not None:
                n += 1
                heapq.heappush(inflight, (res[1], n, res[0], t))
        while inflight and inflight[0][0] <= t:                # DELIVER
            due, _, data, t_sent = heapq.heappop(inflight)
            link.delivered(len(data), due - t_sent)
            ground.ingest(data, t)
            for hid in ground.hazards:
                first_rx.setdefault(hid, t)
        t += DT

    # ---- metrics -------------------------------------------------------------------------------
    confirmed_truth = delivered_truth = 0
    errs, lat = [], []
    for h in truth:
        near = [tr for tr in hmap.tracks.values() if tr.cls == h["cls"] and tr.hits >= hmap.confirm_hits
                and math.hypot(tr.E - h["E"], tr.N - h["N"]) <= 1.5]
        if not near:
            continue                                           # never confirmed onboard (simulated misses)
        confirmed_truth += 1
        got = [g for g in ground.hazards.values() if g["cls"] == h["cls"]
               and math.hypot(g["E"] - h["E"], g["N"] - h["N"]) <= 2.0]
        if got:
            delivered_truth += 1
            g = min(got, key=lambda g: math.hypot(g["E"] - h["E"], g["N"] - h["N"]))
            errs.append(math.hypot(g["E"] - h["E"], g["N"] - h["N"]))
            if g["hid"] in confirmed_at and g["hid"] in first_rx:
                lat.append(first_rx[g["hid"]] - confirmed_at[g["hid"]])
    st = link.stats
    ground.sock.close()
    lat.sort()
    return {
        "loss_configured": loss, "hazards_in_mission": len(truth), "hazards_confirmed_onboard": confirmed_truth,
        "hazards_delivered": delivered_truth,
        "delivery_ratio": round(delivered_truth / confirmed_truth, 3) if confirmed_truth else None,
        "latency_median_s": round(statistics.median(lat), 1) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
        "latency_max_s": round(max(lat), 1) if lat else None,
        "position_error_mean_m": round(statistics.mean(errs), 2) if errs else None,
        "position_error_max_m": round(max(errs), 2) if errs else None,
        "packets_sent": pkts, "packets_delivered": st["delivered"], "packets_corrupted_in_link": st["corrupted"],
        "packets_rejected_by_crc": ground.bad,
        "loss_measured_by_ground_pct": round(100 * (1 - sum(ground.window) / len(ground.window)), 1) if ground.window else None,
        "bytes_on_air": st["bytes_in"],
        "bytes_per_delivered_hazard": round(st["bytes_in"] / delivered_truth) if delivered_truth else None,
        "hazard_records_sent": hz_sent,
    }


# ------------------------------------------------------------------------------------ pytest
def test_clean_link_delivers_everything():
    r = run_scenario(0.0, blackout=None)
    assert r["hazards_confirmed_onboard"] >= 15
    assert r["delivery_ratio"] == 1.0
    assert r["packets_rejected_by_crc"] == 0


def test_run_is_deterministic():
    assert run_scenario(0.3) == run_scenario(0.3)


def test_corrupted_packets_never_reach_the_map():
    r = run_scenario(0.1, corrupt=0.3)
    assert r["packets_corrupted_in_link"] > 0
    assert r["packets_rejected_by_crc"] > 0                    # CRC caught them
    assert r["position_error_max_m"] is not None and r["position_error_max_m"] < 3.0   # no garbage displayed


def test_survival_under_burst_loss_with_blackout():
    for loss, floor in ((0.1, 0.9), (0.3, 0.8)):
        assert run_scenario(loss)["delivery_ratio"] >= floor


if __name__ == "__main__":
    levels = (0.0, 0.1, 0.3, 0.6)
    rows = [run_scenario(l) for l in levels]
    out = ROOT / "test_outputs"
    out.mkdir(exist_ok=True)
    (out / "delivery_report.json").write_text(json.dumps(
        {"scheme": "baseline: self-contained packets re-sent every second, no ACK",
         "link": {**LINK, "burst_mean_packets": LINK["burst"], "blackout_s": [120, 135]}, "results": rows}, indent=2))
    print(f"{'loss':>5} {'delivered':>10} {'ratio':>6} {'lat med/p95/max (s)':>22} {'pos err mean/max (m)':>22} {'bytes/hazard':>13}")
    for r in rows:
        print(f"{r['loss_configured']:>5.0%} {r['hazards_delivered']:>4}/{r['hazards_confirmed_onboard']:<5} {r['delivery_ratio']:>6} "
              f"{str((r['latency_median_s'], r['latency_p95_s'], r['latency_max_s'])):>22} "
              f"{str((r['position_error_mean_m'], r['position_error_max_m'])):>22} {r['bytes_per_delivered_hazard']:>13}")
