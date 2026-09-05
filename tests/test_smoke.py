"""Scaffold smoke test (T0).

Exists so that `make test` has something to collect on an otherwise empty
suite. Real coverage starts at T1.
"""

import src


def test_package_imports() -> None:
    assert src.__name__ == "src"
