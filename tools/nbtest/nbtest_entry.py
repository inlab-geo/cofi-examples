"""Entry point used by the ./nbtest wrapper.

Puts this folder on sys.path so `cofi_nbtest` is importable without installing it and without
touching PYTHONPATH (which images may set themselves).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cofi_nbtest.cli import main  # noqa: E402

sys.exit(main())
