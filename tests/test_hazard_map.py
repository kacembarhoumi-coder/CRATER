"""Unit tests for crater.hazard_map (merging repeated sightings, memory bounding, priority)."""
import math
import pytest
from crater.hazard_map import HazardMap, is_hazard


def test_is_hazard_filtering():
    hf = {"rock_min_height_m": 0.1, "rock_min_width_m": 0.2, "crater_min_width_m": 0.3, "rock_rule": "all", "max_range_m": 5.0}
    # rock_rule == "all": must be BOTH wide enough AND tall enough
    assert is_hazard("rock", 0.25, 0.15, hf) is True   # wide AND tall enough -> hazard
    assert is_hazard("rock", 0.25, 0.05, hf) is False  # wide but flat -> can roll over
    assert is_hazard("rock", 0.10, 0.15, hf) is False  # tall but narrow -> small pebble
    assert is_hazard("rock", 0.10, 0.05, hf) is False  # small rock -> safe regolith
    # craters: must meet minimum width
    assert is_hazard("crater", 0.35, None, hf) is True
    assert is_hazard("crater", 0.1, None, hf) is False
    # unmeasured width: cannot be verified as hazard
    assert is_hazard("rock", None, None, hf) is False
    # range limit (trusted distance <= 5.0m)
    assert is_hazard("rock", 0.3, 0.2, hf, distance_m=3.0) is True
    assert is_hazard("rock", 0.3, 0.2, hf, distance_m=7.0) is False


def test_hazard_map_tracks_and_merges():
    hmap = HazardMap()
    det1 = {"cls": "rock", "E": 5.0, "N": 2.0, "unc95": 0.5, "conf": 0.9, "width": 0.4}
    det2 = {"cls": "rock", "E": 5.1, "N": 2.05, "unc95": 0.3, "conf": 0.95, "width": 0.4}

    hmap.update([det1], t=1.0)
    assert len(hmap.tracks) == 1
    tr_id = list(hmap.tracks.keys())[0]
    assert hmap.tracks[tr_id].hits == 1

    hmap.update([det2], t=2.0)
    assert len(hmap.tracks) == 1
    tr = hmap.tracks[tr_id]
    assert tr.hits == 2
    # position updated with inverse variance weighting: det2 with smaller unc95 pulled E closer to 5.1
    assert 5.05 < tr.E < 5.10
    assert tr.unc95 < 0.3  # uncertainty decreased after fusion


def test_hazard_map_bounded_memory():
    hmap = HazardMap(max_tracks=3)
    for i in range(5):
        hmap.update([{"cls": "rock", "E": 10.0 * i, "N": 0.0, "unc95": 0.5, "conf": 0.8, "width": 0.5}], t=float(i))
    assert len(hmap.tracks) <= 3


def test_packet_hazards_confirmation_and_priority():
    hmap = HazardMap(confirm_hits=2)
    # Add a hazard seen 3 times
    for t in (1.0, 2.0, 3.0):
        hmap.update([{"cls": "rock", "E": 3.0, "N": 0.0, "unc95": 0.3, "conf": 0.9, "width": 0.6}], t=t)

    # Add another hazard seen only once (unconfirmed)
    hmap.update([{"cls": "crater", "E": 4.0, "N": 1.0, "unc95": 0.4, "conf": 0.8, "width": 0.8}], t=3.0)

    # Should only return the confirmed rock
    packet_hzs = hmap.packet_hazards(rE=0.0, rN=0.0, t=3.0, limit=5)
    assert len(packet_hzs) == 1
    assert packet_hzs[0].cls == "rock"
