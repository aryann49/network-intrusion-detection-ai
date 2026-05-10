"""
Compatibility wrapper for the main dashboard.

Prefer running the project from the repository root:
    python app.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import main


if __name__ == "__main__":
    main()
