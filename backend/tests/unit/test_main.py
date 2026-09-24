from __future__ import annotations

import importlib

import pytest

import grounded.main
from grounded.settings import get_settings


def test_importing_the_app_module_reads_no_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # A broken developer .env/env must not break test collection or tooling that imports the module;
    # settings are read only when the app factory runs.
    monkeypatch.setenv("APP_ENV", "not-an-env")
    get_settings.cache_clear()
    try:
        importlib.reload(grounded.main)
    finally:
        get_settings.cache_clear()
