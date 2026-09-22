"""Thresholds that decide where a detection goes: accept, ask Jev, or escalate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    #: top-1 confidence at or above this is accepted from the detector alone.
    high_confidence: float = 0.85

    #: top-1 confidence below this is too weak to disambiguate at all;
    #: skip Jev and escalate directly (garbage in, garbage out).
    ambiguous_floor: float = 0.15

    #: Jev's own answer confidence must clear this to be trusted; below it,
    #: the detection still escalates even after asking.
    jev_confidence_floor: float = 0.5

    #: how many ranked candidates to request from the detector per box.
    top_k: int = 3

    #: free-text scene/application context handed to Jev alongside each
    #: batch of questions (e.g. "camera mounted in a kennel; breeds only").
    context: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.ambiguous_floor <= self.high_confidence <= 1.0:
            raise ValueError(
                "expected 0 <= ambiguous_floor <= high_confidence <= 1, got "
                f"ambiguous_floor={self.ambiguous_floor}, high_confidence={self.high_confidence}"
            )
        if not 0.0 <= self.jev_confidence_floor <= 1.0:
            raise ValueError("jev_confidence_floor must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
