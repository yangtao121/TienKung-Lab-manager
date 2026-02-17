#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path


def _resolve_isaaclab_root() -> Path:
    env_root = os.environ.get("ISAACLAB_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "IsaacLab").resolve()


def main() -> None:
    # Trigger gym task registration on import.
    import tienkung_manager_lab  # noqa: F401

    isaaclab_root = _resolve_isaaclab_root()
    train_script = isaaclab_root / "scripts" / "reinforcement_learning" / "skrl" / "train.py"

    if not train_script.is_file():
        raise FileNotFoundError(
            "Could not find IsaacLab skrl train script at "
            f"'{train_script}'. Set ISAACLAB_ROOT to your IsaacLab directory."
        )

    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
