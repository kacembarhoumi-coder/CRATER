"""
telemetry_packet.py — ultra-compact binary telemetry for the C.R.A.T.E.R rover.

Packet = 18 bytes fixed + 7 bytes per hazard (a 5-hazard packet is 53 bytes).

  HEADER (8 B)        ver|type (1)  rover_id (1)  seq (2)  time in 0.1 s (4)
  ROVER STATE (7 B)   E dm (2)  N dm (2)  yaw 0-255 (1)  pose sigma dm (1)  flags (1)
  COUNT (1 B)         number of hazards
  HAZARD (7 B each)   id(12 bit)|class(4 bit) (2)  dE dm (1)  dN dm (1)
                      uncertainty dm (1)  confidence(4 bit)|width(4 bit) (1)  age s (1)
  CRC-16 (2 B)        CRC-16/CCITT over everything before it

All multi-byte fields are little-endian. Positions are rounded to 0.1 m because the
localization error is already >= 6 cm: we never send precision we don't have.

Dependencies: Python standard library only.
"""
import binascii
import math
import struct
from dataclasses import dataclass

VERSION = 1
TYPE_TELEMETRY = 1
TYPE_HEARTBEAT = 2

CLASSES = ["rock", "crater"]            # index = 4-bit class code
FLAG_HAZARD_AHEAD = 0x01                # example status bits
FLAG_LOW_BATTERY = 0x02
FLAG_BLACKOUT_RECOVERY = 0x04

_HEAD = struct.Struct("<BBHI")          # ver|type, rover_id, seq, time_ds        = 8 B
_STATE = struct.Struct("<hhBBB")        # E_dm, N_dm, yaw, pose_sigma_dm, flags   = 7 B
_COUNT = struct.Struct("<B")            #                                          = 1 B
_HAZ = struct.Struct("<HbbBBB")         # id|cls, dE, dN, unc, conf|width, age     = 7 B
_CRC = struct.Struct("<H")              #                                          = 2 B

FIXED_SIZE = _HEAD.size + _STATE.size + _COUNT.size + _CRC.size   # 18
HAZARD_SIZE = _HAZ.size                                           # 7


class PacketError(ValueError):
    """Raised when a received packet is corrupted, truncated or unknown."""


@dataclass
class RoverState:
    t: float                 # mission time, seconds since landing
    E: float                 # meters East of the lander
    N: float                 # meters North of the lander
    yaw_deg: float           # heading, counter-clockwise from East
    pose_sigma: float = 0.0  # 1-sigma position uncertainty, meters
    flags: int = 0


@dataclass
class Hazard:
    hid: int                 # persistent hazard ID (0-4095), same rock = same ID
    cls: str                 # "rock" or "crater"
    E: float                 # absolute position, meters from the lander
    N: float
    unc95: float             # 95% error-ellipse semi-major axis, meters
    conf: float              # detector confidence 0-1
    width: float             # estimated width, meters
    age: float = 0.0         # seconds since this hazard was last seen


# ----------------------------------------------------------------------------- helpers
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _crc16(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)          # CRC-16/CCITT-FALSE


def _dm(meters: float) -> int:
    return int(round(meters * 10))


# ----------------------------------------------------------------------------- encode
def encode_packet(seq: int, rover_id: int, state: RoverState, hazards=()) -> bytes:
    """Build one self-contained packet. Hazards that cannot be encoded (too far from
    the rover for a 1-byte offset) are skipped, never silently corrupted."""
    ptype = TYPE_TELEMETRY if hazards else TYPE_HEARTBEAT
    head = _HEAD.pack((VERSION << 4) | ptype, rover_id & 0xFF, seq & 0xFFFF,
                      int(round(state.t * 10)) & 0xFFFFFFFF)
    yaw_byte = int(round((state.yaw_deg % 360) / 360 * 256)) % 256
    body = _STATE.pack(_clamp(_dm(state.E), -32768, 32767), _clamp(_dm(state.N), -32768, 32767),
                       yaw_byte, _clamp(_dm(state.pose_sigma), 0, 255), state.flags & 0xFF)

    recs = []
    for h in hazards:
        # offset from the QUANTIZED rover position, so the decoded hazard is within 5 cm
        dE, dN = _dm(h.E) - _dm(state.E), _dm(h.N) - _dm(state.N)
        if not (-128 <= dE <= 127 and -128 <= dN <= 127):
            continue                                       # beyond +-12.7 m: not encodable
        id_cls = ((h.hid & 0x0FFF) << 4) | (CLASSES.index(h.cls) & 0x0F)
        conf4 = _clamp(int(round(h.conf * 15)), 0, 15)
        width4 = _clamp(_dm(h.width), 0, 15)               # 0-1.5 m in 0.1 m steps
        recs.append(_HAZ.pack(id_cls, dE, dN, _clamp(_dm(h.unc95), 0, 255),
                              (conf4 << 4) | width4, _clamp(int(round(h.age)), 0, 255)))
        if len(recs) == 255:
            break

    payload = head + body + _COUNT.pack(len(recs)) + b"".join(recs)
    return payload + _CRC.pack(_crc16(payload))


# ----------------------------------------------------------------------------- decode
def decode_packet(data: bytes) -> dict:
    """Parse and verify a packet. Raises PacketError if it is truncated or corrupted."""
    if len(data) < FIXED_SIZE:
        raise PacketError(f"too short: {len(data)} B")
    payload, (crc,) = data[:-2], _CRC.unpack(data[-2:])
    if _crc16(payload) != crc:
        raise PacketError("CRC mismatch (corrupted packet)")

    ver_type, rover_id, seq, t_ds = _HEAD.unpack_from(payload, 0)
    if ver_type >> 4 != VERSION:
        raise PacketError(f"unknown version {ver_type >> 4}")
    off = _HEAD.size
    e_dm, n_dm, yaw_b, sig_dm, flags = _STATE.unpack_from(payload, off)
    off += _STATE.size
    (count,) = _COUNT.unpack_from(payload, off)
    off += _COUNT.size
    if len(payload) != off + count * HAZARD_SIZE:
        raise PacketError("length does not match hazard count")

    rover = {"E": e_dm / 10, "N": n_dm / 10, "yaw_deg": yaw_b * 360 / 256,
             "pose_sigma": sig_dm / 10, "flags": flags}
    hazards = []
    for _ in range(count):
        id_cls, dE, dN, unc, cw, age = _HAZ.unpack_from(payload, off)
        off += HAZARD_SIZE
        cls_code = id_cls & 0x0F
        hazards.append({
            "hid": id_cls >> 4,
            "cls": CLASSES[cls_code] if cls_code < len(CLASSES) else f"class{cls_code}",
            "E": round(rover["E"] + dE / 10, 1),           # back to absolute coordinates
            "N": round(rover["N"] + dN / 10, 1),
            "unc95": unc / 10,
            "conf": (cw >> 4) / 15,
            "width": (cw & 0x0F) / 10,
            "age": age,
        })
    return {"type": "telemetry" if ver_type & 0x0F == TYPE_TELEMETRY else "heartbeat",
            "rover_id": rover_id, "seq": seq, "t": t_ds / 10, "rover": rover, "hazards": hazards,
            "size_bytes": len(data)}


def max_hazards_for(budget_bytes: int) -> int:
    """How many hazards fit in a packet of at most budget_bytes."""
    return max(0, (budget_bytes - FIXED_SIZE) // HAZARD_SIZE)
