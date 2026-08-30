from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


def bootstrap_project(
    project_root: str | Path,
    *,
    install_dependencies: bool = False,
    install_rl_dependencies: bool = False,
) -> Path:
    """Make the src-layout package importable in notebooks and Colab runtimes."""
    root = Path(project_root).expanduser().resolve()
    package_init = root / "src" / "gas_portfolio_rl" / "__init__.py"
    pyproject = root / "pyproject.toml"
    if not package_init.is_file() or not pyproject.is_file():
        raise FileNotFoundError(
            "Некорректный PROJECT_ROOT. Ожидались файлы "
            f"{package_init} и {pyproject}."
        )

    source_path = str(root / "src")
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    importlib.invalidate_caches()

    loaded_module = sys.modules.get("gas_portfolio_rl")
    if loaded_module is not None:
        loaded_path = Path(loaded_module.__file__).resolve()
        if root not in loaded_path.parents:
            for name in tuple(sys.modules):
                if name == "gas_portfolio_rl" or name.startswith("gas_portfolio_rl."):
                    del sys.modules[name]

    if install_dependencies:
        target = f"{root}[rl]" if install_rl_dependencies else str(root)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-build-isolation",
                "-e",
                target,
            ]
        )
        importlib.invalidate_caches()

    module = importlib.import_module("gas_portfolio_rl")
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise ImportError(
            "Импортирован другой gas_portfolio_rl: "
            f"{module_path}. Ожидался пакет из {root}."
        )
    return root
