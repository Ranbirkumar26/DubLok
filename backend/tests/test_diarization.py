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


def _words(text: str, start: float, step: float = 0.3):
    return [
        {"start": start + i * step, "end": start + (i + 1) * step - 0.02, "word": f" {w}"}
        for i, w in enumerate(text.split())
    ]


def test_split_by_turns_splits_on_speaker_change_and_flags_overlap():
    from app.pipeline.diarization import split_segments_by_turns

    words = _words("where is it over there no wait I said", 0.0)
    segment = Segment(id="s", speaker="Speaker 1", start=0.0, end=2.7, text="x", words=words)
    turns = [(0.0, 1.5, "A"), (1.2, 2.8, "B")]  # B interrupts A: 1.2-1.5 overlap
    result = split_segments_by_turns([segment], turns)
    assert [s.speaker for s in result] == ["Speaker 1", "Speaker 2"]
    assert result[0].overlap and not result[1].overlap
    assert result[0].text.startswith("where is it")
    assert result[0].end <= result[1].start + 0.3
    assert [s.id for s in result] == ["seg_0001", "seg_0002"]


def test_split_by_turns_single_word_does_not_make_a_segment():
    from app.pipeline.diarization import split_segments_by_turns

    words = _words("hello there my friend ok", 0.0)
    segment = Segment(id="s", speaker="Speaker 1", start=0.0, end=1.5, text="x", words=words)
    result = split_segments_by_turns([segment], [(0.0, 1.2, "A"), (1.2, 1.6, "B")])
    assert len(result) == 1 and result[0].text.endswith("ok")
