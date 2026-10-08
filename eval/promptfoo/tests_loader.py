"""promptfoo test generator: a shim over ``grounded.evals.promptfoo_tests`` (Tech §15.3).

``EVAL_QUESTION_IDS`` (golden ids separated by commas or spaces, e.g. ``q003,q045``) limits a run to
those questions for a smoke run; unset, the whole golden set runs. It is read here, in the loader's
own process, and nowhere else: the backend package never sees it.
"""

import os

from grounded.evals.promptfoo_tests import generate_tests as _generate_tests
from grounded.evals.promptfoo_tests import parse_question_ids


def generate_tests(config=None):
    """Called by promptfoo (``tests: file://tests_loader.py:generate_tests``)."""
    options = dict(config or {})
    ids = parse_question_ids(os.getenv("EVAL_QUESTION_IDS"))
    if ids is not None:
        options["ids"] = ids
    return _generate_tests(options)
