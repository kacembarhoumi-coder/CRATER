"""Unit tests for crater.ground (GroundStation & packet ingestion)."""
import time
import pytest
import socket
from crater.ground import GroundStation
from crater.telemetry import encode_packet, RoverState, Hazard


class MockLink:
    def snapshot(self):
        return {
            "last_delay": 0.04,
            "blackout": False,
            "dropped_loss": 0,
            "dropped_blackout": 0,
            "dropped_queue": 0,
            "params": {"loss": 0.1}
        }


def test_ground_station_ingestion():
    ground_port = 9301
    link = MockLink()
    gs = GroundStation(ground_port, link)
    gs.start()

    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state = RoverState(t=10.0, E=5.0, N=2.0, yaw_deg=15.0, pose_sigma=0.05)
        hz = [Hazard(hid=101, cls="rock", E=8.0, N=2.0, unc95=0.4, conf=0.9, width=0.5, age=0.0)]
        pkt = encode_packet(seq=1, rover_id=1, state=state, hazards=hz)

        sender.sendto(pkt, ("127.0.0.1", ground_port))
        time.sleep(0.3)

        snap = gs.snapshot()
        assert snap["rover"]["E"] == 5.0
        assert snap["rover"]["N"] == 2.0
        assert len(snap["hazards"]) == 1
        assert snap["hazards"][0]["hid"] == 101
        assert snap["link"]["rx"] == 1
        assert snap["link"]["status"] == "OK"
    finally:
        gs.stop()
