"""promptfoo Python provider: a shim over ``grounded.evals.promptfoo_provider`` (Tech §15.3).

The logic lives in the package, where ruff, pyright and pytest cover it. This file only has to
exist next to the config, because promptfoo loads ``file://provider.py`` and calls ``call_api``.

Eval mode is set here, before ``grounded.settings`` is first read, because the process promptfoo
starts has the shell's environment and ``.env``, not the eval's: ``APP_ENV`` is forced to ``eval``
and ``GENERATOR_PROVIDERS`` defaults to ``gemini`` (eval mode refuses the default two-provider
list). Export ``GENERATOR_PROVIDERS=fake`` first for a smoke run with the canned stub and no
generator key. This is the one place outside ``grounded.settings`` that touches the environment: it
writes, it never reads a setting.
"""

import os

os.environ["APP_ENV"] = "eval"
os.environ.setdefault("GENERATOR_PROVIDERS", "gemini")

from grounded.evals.promptfoo_provider import call_api as call_api  # noqa: E402
