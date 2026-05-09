"""Unit tests for IdentityResolver.

Voting: per-track most-common (colour, number) ≥ min_votes wins.
Merging: tracks that resolve to the same (colour, number) collapse to canonical.
Lookup: follows merge links so callers always get the canonical identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.identity import IdentityResolver
from src.types import OCRReading


def _r(track_id: int, colour: str, number: int | None, frame: int = 0) -> OCRReading:
    return OCRReading(
        track_id=track_id,
        frame_idx=frame,
        colour=colour,  # type: ignore[arg-type]
        number=number,
        ocr_confidence=0.9,
    )


def test_unanimous_readings_resolve_track():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for f in range(5):
        res.add_reading(_r(track_id=1, colour="red", number=7, frame=f))
    out = res.resolve()
    assert out[1].bib_id == "R7"
    assert out[1].colour == "red"
    assert out[1].number == 7
    assert out[1].vote_count == 5


def test_majority_vote_wins_over_noise():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(7):
        res.add_reading(_r(1, "red", 7))
    for _ in range(2):
        res.add_reading(_r(1, "red", 4))  # noise
    out = res.resolve()
    assert out[1].bib_id == "R7"
    assert out[1].vote_count == 7


def test_below_min_votes_remains_unresolved():
    res = IdentityResolver(Config(min_votes_for_identity=3))
    res.add_reading(_r(1, "red", 7))
    res.add_reading(_r(1, "red", 7))  # only 2 votes, threshold is 3
    out = res.resolve()
    assert out[1].bib_id is None
    assert out[1].colour == "unknown"


def test_unknown_colour_readings_excluded():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "unknown", None))
    out = res.resolve()
    assert out[1].bib_id is None
    assert out[1].sample_count == 5
    assert out[1].vote_count == 0


def test_high_bib_numbers_resolve():
    # OCR range is 1..99, so number 99 is valid and should resolve.
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "red", 99))
    out = res.resolve()
    assert out[1].bib_id == "R99"


def test_two_tracks_same_bib_get_merged():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "red", 7))
    for _ in range(3):
        res.add_reading(_r(2, "red", 7))
    out = res.resolve()
    canonical = out[1] if out[1].vote_count >= out[2].vote_count else out[2]
    other = out[2] if canonical.track_id == 1 else out[1]
    assert canonical.merged_into is None
    assert other.merged_into == canonical.track_id


def test_lookup_follows_merge_link():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "red", 7))
    for _ in range(3):
        res.add_reading(_r(2, "red", 7))
    res.resolve()
    assert res.lookup(1).bib_id == "R7"
    assert res.lookup(2).bib_id == "R7"
    # Both lookups return the same canonical identity object's bib_id
    assert res.lookup(1).track_id == res.lookup(2).track_id


def test_lookup_unknown_track_returns_none():
    res = IdentityResolver(Config())
    assert res.lookup(999) is None


def test_resolve_is_idempotent():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "red", 7))
    out1 = res.resolve()
    out2 = res.resolve()
    assert out1[1].bib_id == out2[1].bib_id
    assert out1[1].vote_count == out2[1].vote_count


def test_different_bibs_dont_merge():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "red", 7))
    for _ in range(5):
        res.add_reading(_r(2, "blue", 7))
    out = res.resolve()
    assert out[1].merged_into is None
    assert out[2].merged_into is None
    assert out[1].bib_id == "R7"
    assert out[2].bib_id == "B7"


def test_colour_only_resolves_when_ocr_always_fails():
    # When colour is consistent but OCR never fires, the track should still get
    # a colour-only bib_id (e.g. "P?") rather than staying unresolved.
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(5):
        res.add_reading(_r(1, "pink", None))
    out = res.resolve()
    assert out[1].bib_id == "P?"
    assert out[1].colour == "pink"
    assert out[1].number is None


def test_full_resolution_preferred_over_colour_only():
    # If there are enough (colour, number) readings, use them rather than
    # falling back to colour-only even though colour votes are also present.
    res = IdentityResolver(Config(min_votes_for_identity=2))
    for _ in range(3):
        res.add_reading(_r(1, "green", 5))
    for _ in range(2):
        res.add_reading(_r(1, "green", None))  # colour-only samples
    out = res.resolve()
    assert out[1].bib_id == "G5"
    assert out[1].number == 5


def test_lookup_re_resolves_after_new_reading():
    res = IdentityResolver(Config(min_votes_for_identity=2))
    res.add_reading(_r(1, "red", 7))
    res.add_reading(_r(1, "red", 7))
    assert res.lookup(1).bib_id == "R7"
    # New conflicting readings shouldn't change the winner here, but lookup
    # must not raise on a re-resolve.
    res.add_reading(_r(1, "red", 4))
    assert res.lookup(1).bib_id == "R7"
