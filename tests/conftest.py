"""
tests/conftest.py: Shared fixtures and environment setup for QuakeSnail E2E tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is at front of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return PROJECT_ROOT / "tests"
