"""Unit tests for crater.link (lossy radio link emulator)."""
import time
import pytest
import socket
from crater.link import LinkEmulator


def test_link_packet_transmission_and_stats():
    # Ports for testing
    listen_port = 9201
    dest_port = 9202

    dest_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_sock.bind(("127.0.0.1", dest_port))
    dest_sock.settimeout(0.5)

    link = LinkEmulator(("127.0.0.1", listen_port), ("127.0.0.1", dest_port), loss=0.0, delay_ms=10, jitter_ms=0, bandwidth_bps=100000)
    link.start()

    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = b"test_payload_packet"
        sender.sendto(payload, ("127.0.0.1", listen_port))

        data, _ = dest_sock.recvfrom(1024)
        assert data == payload

        st = link.snapshot()
        assert st["sent"] == 1
        assert st["delivered"] == 1
        assert st["dropped_loss"] == 0
    finally:
        link.stop()
        dest_sock.close()


def test_link_blackout():
    listen_port = 9203
    dest_port = 9204

    dest_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_sock.bind(("127.0.0.1", dest_port))
    dest_sock.settimeout(0.2)

    link = LinkEmulator(("127.0.0.1", listen_port), ("127.0.0.1", dest_port), loss=0.0)
    link.start()
    link.blackout(5.0)

    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b"dropped_packet", ("127.0.0.1", listen_port))

        with pytest.raises(socket.timeout):
            dest_sock.recvfrom(1024)

        st = link.snapshot()
        assert st["blackout"] is True
        assert st["dropped_blackout"] == 1
    finally:
        link.stop()
        dest_sock.close()
