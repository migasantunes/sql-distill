"""Invariant 6: the whole package imports without any heavy dependency."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("torch", "transformers", "peft", "anthropic", "gradio", "bitsandbytes")

SCRIPT = f"""
import importlib, pkgutil, sys
import sqldistill
names = [m.name for m in pkgutil.walk_packages(sqldistill.__path__, "sqldistill.")]
for name in names:
    importlib.import_module(name)
assert len(names) > 10, names
loaded = sorted(m for m in {HEAVY!r} if m in sys.modules)
assert not loaded, f"heavy modules imported: {{loaded}}"
print(len(names))
"""


def test_every_submodule_imports_without_heavy_deps() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
