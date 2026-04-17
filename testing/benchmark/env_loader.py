from __future__ import annotations

import os
from pathlib import Path


_ENV_LOADED = False


def load_project_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    project_root = Path(__file__).resolve().parents[2]
    env_paths = [
        project_root / ".env",
        project_root / ".env.local",
        project_root / "backend" / ".env",
    ]

    for env_path in env_paths:
        if not env_path.is_file():
            continue

        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key or key in os.environ:
                continue

            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ[key] = value
