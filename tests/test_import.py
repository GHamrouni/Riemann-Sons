"""Importing a library must preserve the application's numerical defaults."""

import subprocess
import sys


def test_import_preserves_default_dtype():
    subprocess.run(
        [sys.executable, "-c", "import torch; import riemann_and_sons; assert torch.get_default_dtype() == torch.float32"],
        check=True,
        capture_output=True,
        text=True,
    )
