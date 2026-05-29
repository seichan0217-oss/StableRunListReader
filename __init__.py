"""Tools for building AHCAL stable run lists."""

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from reader import StableRunListReader

__all__ = ["StableRunListReader"]
