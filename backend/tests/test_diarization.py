from __future__ import annotations

import numpy as np

from app.pipeline.diarization import assign_two_segment_speakers
from app.pipeline.diarization import Diarizer
from app.pipeline.types import Segment
from app.config import Settings


def _segments() -> list[Segment]:
    return [
        Segment(id="seg_0001", speaker="Speaker 1", start=0.0, end=1.0, text="Question"),
        Segment(id="seg_0002", speaker="Speaker 1", start=1.5, end=2.5, text="Answer"),
    ]


def test_two_segment_diarization_splits_different_embeddings() -> None:
    segments = _segments()

    assign_two_segment_speakers(
        segments,
        [0, 1],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    )

    assert [segment.speaker for segment in segments] == ["Speaker 1", "Speaker 2"]


def test_two_segment_diarization_keeps_similar_embeddings() -> None:
    segments = _segments()

    assign_two_segment_speakers(
        segments,
        [0, 1],
        [np.array([1.0, 0.0]), np.array([0.99, 0.01])],
    )

    assert [segment.speaker for segment in segments] == ["Speaker 1", "Speaker 1"]


def test_short_dialogue_fallback_splits_two_paused_turns() -> None:
    segments = _segments()

    result = Diarizer(Settings())._short_dialogue_fallback(segments)

    assert [segment.speaker for segment in result] == ["Speaker 1", "Speaker 2"]


def test_short_dialogue_fallback_keeps_continuous_segments() -> None:
    segments = [
        Segment(id="seg_0001", speaker="Speaker 1", start=0.0, end=1.0, text="First"),
        Segment(id="seg_0002", speaker="Speaker 1", start=1.1, end=2.0, text="Second"),
    ]

    result = Diarizer(Settings())._short_dialogue_fallback(segments)

    assert [segment.speaker for segment in result] == ["Speaker 1", "Speaker 1"]


def test_pause_heuristic_does_not_override_embedding_decision(monkeypatch) -> None:
    diarizer = Diarizer(Settings(engine_mode="production"))
    monkeypatch.setattr(diarizer, "_speechbrain_cluster", lambda _audio, segments: segments)

    result = diarizer.assign_speakers(None, _segments())

    assert [segment.speaker for segment in result] == ["Speaker 1", "Speaker 1"]


def test_pause_heuristic_used_when_embeddings_fail(monkeypatch) -> None:
    diarizer = Diarizer(Settings(engine_mode="production"))

    def boom(_audio, _segments):
        raise OSError("symlink privilege not held")

    monkeypatch.setattr(diarizer, "_speechbrain_cluster", boom)

    result = diarizer.assign_speakers(None, _segments())

    assert [segment.speaker for segment in result] == ["Speaker 1", "Speaker 2"]


def test_short_segments_inherit_the_nearest_longer_speaker() -> None:
    from app.pipeline.diarization import inherit_short_segment_speakers

    segments = [
        Segment(id="a", speaker="Speaker 1", start=0, end=4, text="x"),
        Segment(id="b", speaker="Speaker 1", start=4, end=5, text="y"),
        Segment(id="c", speaker="Speaker 2", start=5, end=9, text="z"),
        Segment(id="d", speaker="Speaker 1", start=9, end=10, text="w"),
    ]
    inherit_short_segment_speakers(segments, [0, 2])

    assert [segment.speaker for segment in segments] == ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 2"]
