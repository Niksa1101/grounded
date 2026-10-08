"""promptfoo Python assertions: a shim over ``grounded.evals.promptfoo_asserts`` (Tech §15.3).

promptfoo starts a new Python process for every assertion call, so keep this file to the
environment and the imports.

The two judge assertions (``faithfulness``, ``correctness``) read the settings in this process, so
it gets the same eval environment as the provider shim: ``APP_ENV`` is forced to ``eval`` and
``GENERATOR_PROVIDERS`` defaults to ``gemini``. Export ``GENERATOR_PROVIDERS=fake`` first for a run
with no network: the judge is then the canned stub, too. Like ``provider.py``, this only writes the
environment, before ``grounded.settings`` is first read.
"""

import os

os.environ["APP_ENV"] = "eval"
os.environ.setdefault("GENERATOR_PROVIDERS", "gemini")

from grounded.evals.promptfoo_asserts import citation_precision as citation_precision  # noqa: E402
from grounded.evals.promptfoo_asserts import citation_validity as citation_validity  # noqa: E402
from grounded.evals.promptfoo_asserts import correctness as correctness  # noqa: E402
from grounded.evals.promptfoo_asserts import faithfulness as faithfulness  # noqa: E402
from grounded.evals.promptfoo_asserts import refusal_correctness as refusal_correctness  # noqa: E402
from grounded.evals.promptfoo_asserts import schema_first_try as schema_first_try  # noqa: E402
