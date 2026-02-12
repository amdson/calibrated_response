from __future__ import annotations

import os

# Work around pgmax/numba caching issue in this environment.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
