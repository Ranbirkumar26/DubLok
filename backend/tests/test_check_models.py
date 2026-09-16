from __future__ import annotations

from scripts import check_models


def test_production_preflight_fails_when_model_packages_are_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_models, "_setting", lambda name, default="": "production")
    monkeypatch.setattr(check_models.shutil, "which", lambda _command: "found")
    monkeypatch.setattr(check_models.importlib.util, "find_spec", lambda _package: None)

    assert check_models.main() == 1
    assert "Install missing production Python packages" in capsys.readouterr().out


def test_demo_preflight_allows_missing_model_packages(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_models, "_setting", lambda name, default="": "demo")
    monkeypatch.setattr(check_models.shutil, "which", lambda _command: "found")
    monkeypatch.setattr(check_models.importlib.util, "find_spec", lambda _package: None)

    assert check_models.main() == 0
    assert "Engine mode: demo" in capsys.readouterr().out
