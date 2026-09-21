from __future__ import annotations

import pytest

from app.errors import ValidationError
from app.pipeline.runner import validate_translated_segments
from app.pipeline.types import Segment


def test_validate_translated_segments_rejects_empty_translation() -> None:
    segments = [
        Segment(
            id="seg_0001",
            speaker="Speaker 1",
            start=0.0,
            end=1.0,
            text="Hello",
            translated_text="",
        )
    ]

    with pytest.raises(ValidationError, match="seg_0001"):
        validate_translated_segments(segments)


def test_validate_translated_segments_allows_reviewed_translation() -> None:
    validate_translated_segments(
        [
            Segment(
                id="seg_0001",
                speaker="Speaker 1",
                start=0.0,
                end=1.0,
                text="Hello",
                translated_text="नमस्ते",
            )
        ]
    )
