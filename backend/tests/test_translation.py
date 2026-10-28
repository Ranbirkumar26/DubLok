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
        "_fallback_translate",
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


def test_translation_uses_nllb_when_indictrans2_is_not_cached(monkeypatch) -> None:
    translator = Translator(Settings(engine_mode="production"))

    monkeypatch.setattr(translator, "_model_is_cached", lambda _model_name: False)
    monkeypatch.setattr(
        translator,
        "_indictrans2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip")),
    )
    monkeypatch.setattr(
        translator,
        "_fallback_translate",
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

    monkeypatch.setattr(translator, "_model_is_cached", lambda _model_name: True)
    monkeypatch.setattr(
        translator,
        "_indictrans2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessingError("gated")),
    )
    monkeypatch.setattr(
        translator,
        "_fallback_translate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessingError("missing")),
    )

    with pytest.raises(ProcessingError, match="Local fallback also failed"):
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


def test_translation_translates_each_sentence_and_rejoins(monkeypatch) -> None:
    translator = Translator(Settings(engine_mode="production"))
    seen: list[list[str]] = []

    def fake_translate(texts, source_language, target_language):
        seen.append(list(texts))
        return [f"<{text}>" for text in texts]

    monkeypatch.setattr(translator, "_model_is_cached", lambda _model_name: False)
    monkeypatch.setattr(translator, "_fallback_translate", fake_translate)

    first, second = translator.translate_segments(
        [
            Segment(id="seg_0001", speaker="Speaker 1", start=0.0, end=3.0,
                    text="body has a turn back moment. You can go forward?"),
            Segment(id="seg_0002", speaker="Speaker 1", start=3.0, end=4.0, text="Never quit"),
        ],
        "en",
        "hi",
    )

    assert seen == [["body has a turn back moment.", "You can go forward?", "Never quit"]]
    assert first.translated_text == "<body has a turn back moment.> <You can go forward?>"
    assert second.translated_text == "<Never quit>"
