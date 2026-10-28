from __future__ import annotations

from app.pipeline.types import Segment, merge_sentence_fragments


def _seg(start: float, end: float, text: str, speaker: str = "Speaker 1") -> Segment:
    return Segment(id="x", speaker=speaker, start=start, end=end, text=text)


def test_merge_joins_mid_sentence_splits_within_limits() -> None:
    merged = merge_sentence_fragments(
        [
            _seg(0.0, 6.08, "You can go forward or you can give up."),
            _seg(6.94, 12.04, "before you give up is that if you give up,"),
            _seg(13.18, 20.66, "the guarantee is it will never happen"),
            _seg(20.66, 28.4, "no way under the sun. if you never"),
            _seg(28.4, 30.64, "give up no matter what."),
        ]
    )

    assert [(s.id, s.start, s.end) for s in merged] == [
        ("seg_0001", 0.0, 6.08),
        ("seg_0002", 6.94, 20.66),
        ("seg_0003", 20.66, 30.64),
    ]
    assert merged[2].text == "no way under the sun. if you never give up no matter what."


def test_merge_keeps_speaker_changes_and_long_gaps() -> None:
    merged = merge_sentence_fragments(
        [
            _seg(0.0, 1.0, "So what is", "Speaker 1"),
            _seg(1.1, 2.0, "my answer", "Speaker 2"),
            _seg(5.0, 6.0, "and then"),
        ]
    )

    assert len(merged) == 3
