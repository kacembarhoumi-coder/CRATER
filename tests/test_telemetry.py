"""Telemetry: compact binary packets, integrity checks, size budget."""
import json
import random

import pytest

from crater.telemetry import (CLASSES, FIXED_SIZE, HAZARD_SIZE, Hazard, PacketError, RoverState,
                              decode_packet, encode_packet, max_hazards_for)

STATE = RoverState(t=1834.27, E=12.43, N=-3.08, yaw_deg=40.2, pose_sigma=0.07, flags=1)


def test_packet_size():
    hz = [Hazard(i, "rock", 14 + i, -2, 0.6, 0.8, 0.4, 1) for i in range(5)]
    assert len(encode_packet(1, 1, STATE, hz)) == FIXED_SIZE + 5 * HAZARD_SIZE == 53
    assert len(encode_packet(2, 1, STATE)) == FIXED_SIZE == 18


def test_round_trip_within_quantization():
    rng = random.Random(0)
    for _ in range(5000):
        s = RoverState(rng.uniform(0, 1e5), rng.uniform(-3000, 3000), rng.uniform(-3000, 3000),
                       rng.uniform(-720, 720))
        h = Hazard(rng.randint(0, 4095), rng.choice(CLASSES), s.E + rng.uniform(-12.6, 12.6),
                   s.N + rng.uniform(-12.6, 12.6), 1.0, 0.5, 0.5, 3)
        d = decode_packet(encode_packet(rng.randint(0, 65535), 3, s, [h]))
        o = d["hazards"][0]
        assert abs(o["E"] - h.E) <= 0.0501 and abs(o["N"] - h.N) <= 0.0501
        assert o["hid"] == h.hid and o["cls"] == h.cls


def test_detects_bit_flip_and_truncation():
    pkt = encode_packet(7, 1, STATE, [Hazard(1, "crater", 14, 0, 1, 0.5, 1, 0)])
    for i in range(len(pkt)):
        bad = bytearray(pkt)
        bad[i] ^= 0x10
        with pytest.raises(PacketError):
            decode_packet(bytes(bad))
    with pytest.raises(PacketError):
        decode_packet(pkt[:-3])


def test_far_hazard_is_skipped_not_corrupted():
    d = decode_packet(encode_packet(1, 1, STATE, [Hazard(5, "rock", STATE.E + 40, STATE.N, 1, 1, 1)]))
    assert d["hazards"] == []


def test_much_smaller_than_json():
    hz = [Hazard(i, "rock", 14 + i * 0.7, -2 + i * 0.5, 0.6, 0.8, 0.4, 1) for i in range(5)]
    js = json.dumps({"rover": vars(STATE), "hazards": [vars(h) for h in hz]}).encode()
    assert len(js) / len(encode_packet(1, 1, STATE, hz)) > 5


def test_budget_helper():
    assert max_hazards_for(64) == 6
    assert max_hazards_for(17) == 0
