"""Bib identity resolver — voting + cross-track merging.

Each tracker ID accumulates `OCRReading`s (sparsely sampled across frames).
After the full video is processed, `resolve()`:
  1. Per track: takes the modal (colour, number) reading where colour is not
     "unknown" and number is not None; requires ≥ `min_votes_for_identity`
     supporting readings to confidently resolve.
  2. Across tracks: when multiple track IDs resolve to the same (colour, number),
     keeps the highest-vote-count track as canonical and marks the others as
     `merged_into = canonical_track_id`.

`lookup(track_id)` walks `merged_into` so callers always get the canonical
identity for any raw tracker ID — used to backfill labels onto every frame
even where OCR didn't fire.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from config import Config
from src.types import BibIdentity, OCRReading

logger = logging.getLogger(__name__)


class IdentityResolver:
    """Per-track voting + cross-track merging for bib identities.

    Args:
        config: Pipeline config; uses `min_votes_for_identity`.
    """

    def __init__(self, config: Config) -> None:
        self._min_votes = config.min_votes_for_identity
        self._readings_by_track: dict[int, list[OCRReading]] = defaultdict(list)
        self._resolved: dict[int, BibIdentity] = {}
        self._dirty = False

    def add_reading(self, reading: OCRReading) -> None:
        """Buffer one OCR sample for later voting."""
        self._readings_by_track[reading.track_id].append(reading)
        self._dirty = True

    def resolve(self) -> dict[int, BibIdentity]:
        """Compute identities by voting + cross-track merging. Idempotent."""
        per_track: dict[int, BibIdentity] = {}
        for track_id, readings in self._readings_by_track.items():
            per_track[track_id] = self._vote(track_id, readings)

        # Cross-track merge: group by (colour, number); keep highest-vote
        groups: dict[tuple[str, int | None], list[BibIdentity]] = defaultdict(list)
        for ident in per_track.values():
            if ident.bib_id is None:
                continue
            groups[(ident.colour, ident.number)].append(ident)

        for group in groups.values():
            if len(group) <= 1:
                continue
            canonical = max(group, key=lambda b: (b.vote_count, -b.track_id))
            for ident in group:
                if ident.track_id != canonical.track_id:
                    ident.merged_into = canonical.track_id

        self._resolved = per_track
        self._dirty = False
        return per_track

    def lookup(self, track_id: int) -> BibIdentity | None:
        """Return the canonical identity for a track_id, following merge links.

        Returns None for tracks with no readings or unresolved identities.
        Re-runs `resolve()` if buffers have changed since last call.
        """
        if self._dirty:
            self.resolve()
        ident = self._resolved.get(track_id)
        if ident is None:
            return None
        if ident.merged_into is not None:
            return self._resolved.get(ident.merged_into, ident)
        return ident

    # Short prefix per colour for the bib_id label (e.g. "P18", "W10")
    _COLOUR_PREFIX: dict[str, str] = {
        "red": "R", "orange": "O", "yellow": "Y", "green": "G",
        "cyan": "C", "blue": "B", "purple": "U", "pink": "P",
        "white": "W", "black": "K",
    }

    def _vote(self, track_id: int, readings: list[OCRReading]) -> BibIdentity:
        sample_count = len(readings)
        colour_reads = [r.colour for r in readings if r.colour != "unknown"]
        full_reads = [
            (r.colour, r.number)
            for r in readings
            if r.colour != "unknown" and r.number is not None
        ]

        if not colour_reads:
            return BibIdentity(
                track_id=track_id,
                bib_id=None,
                colour="unknown",
                number=None,
                vote_count=0,
                sample_count=sample_count,
            )

        # Full resolution: needs a (colour, number) pair with sufficient votes.
        if full_reads:
            (colour, number), vote_count = Counter(full_reads).most_common(1)[0]
            if vote_count >= self._min_votes:
                prefix = self._COLOUR_PREFIX.get(colour, colour[0].upper())
                return BibIdentity(
                    track_id=track_id,
                    bib_id=f"{prefix}{number}",
                    colour=colour,
                    number=number,
                    vote_count=vote_count,
                    sample_count=sample_count,
                )

        # Colour-only fallback — assigns the player to a team even when OCR fails.
        colour, vote_count = Counter(colour_reads).most_common(1)[0]
        if vote_count >= self._min_votes:
            prefix = self._COLOUR_PREFIX.get(colour, colour[0].upper())
            return BibIdentity(
                track_id=track_id,
                bib_id=f"{prefix}?",
                colour=colour,
                number=None,
                vote_count=vote_count,
                sample_count=sample_count,
            )

        return BibIdentity(
            track_id=track_id,
            bib_id=None,
            colour="unknown",
            number=None,
            vote_count=vote_count,
            sample_count=sample_count,
        )
