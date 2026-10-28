from __future__ import annotations

import pytest

from app.config import Settings
from app.errors import ProcessingError
from app.pipeline.translation import Translator
from app.pipeline.types import Segment


def test_translation_uses_nllb_fallback_when_indictrans2_fails(monkeypatch) -> None:
    translator = Translator(Settings(engine_mode="production"))

    monkeypatch.setattr(
        translator,
        "_indictrans2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessingError("gated")),
    )
    monkeypatch.setattr(
        translator,
        "_nllb",
        lambda texts, source_language, target_language: ["namaste" for _ in texts],
    )

    [segment] = translator.translate_segments(
        [
            Segment(
                id="seg_0001",
                speaker="Speaker 1",
                start=0.0,
                end=1.0,
                text="Hello",
            )
        ],
        "en",
        "hi",
    )

    assert segment.translated_text == "namaste"


def test_translation_reports_both_failures_when_fallback_fails(monkeypatch) -> None:
    translator = Translator(Settings(engine_mode="production"))

    monkeypatch.setattr(
        translator,
        "_indictrans2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessingError("gated")),
    )
    monkeypatch.setattr(
        translator,
        "_nllb",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessingError("missing")),
    )

    with pytest.raises(ProcessingError, match="NLLB fallback also failed"):
        translator.translate_segments(
            [
                Segment(
                    id="seg_0001",
                    speaker="Speaker 1",
                    start=0.0,
                    end=1.0,
                    text="Hello",
                )
            ],
            "en",
            "hi",
        )
