"""
secrets.py - Local secret-file helpers for notebook workflows.

This intentionally supports a minimal shell-style format such as:

    export KENPOM_EMAIL="you@example.com"
    export KENPOM_PASSWORD="your-password"
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def load_shell_secrets(
    path: str = ".secrets/kenpom.sh",
    override: bool = False,
) -> dict[str, str]:
    """
    Load simple `export KEY=value` lines from a local shell file into os.environ.

    Returns only the variables loaded from the file during this call.
    Existing environment variables win unless `override=True`.
    """

    secrets_path = Path(path)
    if not secrets_path.exists():
        return {}

    loaded: dict[str, str] = {}

    for raw_line in secrets_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue

        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
        value = tokens[0] if tokens else ""

        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value

    return loaded


def load_kenpom_credentials(
    path: str = ".secrets/kenpom.sh",
    override: bool = False,
) -> tuple[str | None, str | None]:
    """Load the local KenPom secret file, then return the env-backed credentials."""

    load_shell_secrets(path=path, override=override)
    return os.getenv("KENPOM_EMAIL"), os.getenv("KENPOM_PASSWORD")
