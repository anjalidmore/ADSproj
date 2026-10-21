"""Smoke test: every dashboard page renders without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_page_renders(path):
    at = AppTest.from_file(str(path), default_timeout=300).run()
    assert not at.exception, [e.message for e in at.exception]
