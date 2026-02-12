"""Root conftest — ensures AAT's src package is importable."""
from __future__ import annotations

import sys
from pathlib import Path

# Add AAT to sys.path so `from src.rl import ...` resolves to AAT's src package.
# This is necessary because dojo's own package is `dojo/`, not `src/`.
_aat_root = Path(__file__).resolve().parent.parent / "agile-agent-team"
if _aat_root.exists() and str(_aat_root) not in sys.path:
    sys.path.insert(0, str(_aat_root))
