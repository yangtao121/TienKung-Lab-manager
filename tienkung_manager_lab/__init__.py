from __future__ import annotations

import importlib
import pkgutil


# Import only manager-based task modules so gym registration runs on package import.
def _import_manager_based_tasks() -> None:
    package = importlib.import_module(f"{__name__}.manager_based")
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".mdp" in module_name:
            continue
        importlib.import_module(module_name)


_import_manager_based_tasks()

__all__ = []
