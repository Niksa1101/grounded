"""Vercel entry module (Tech §18): the one place that builds the app at import time.

``[tool.vercel] entrypoint = "app:app"`` in ``pyproject.toml`` makes Vercel load the ``app``
variable of this file, and ``vercel.json`` keys the function settings on ``app.py``.

``grounded.main`` stays free of import-time side effects (``create_app`` reads the settings, so it
runs here and not there). The source tree goes first on ``sys.path`` so that the package is the one
in this bundle, whether or not the build also installed it: ``generation/prompts.py`` and
``observability/cost.py`` find ``prompts/`` and ``pricing.toml`` relative to their own files, which
holds for ``backend/src/grounded/`` and not for a copy in ``site-packages``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grounded.main import create_app

app = create_app()
