from pathlib import Path
import sys

import gas_portfolio_rl

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from notebook_bootstrap import bootstrap_project


def test_notebook_bootstrap_imports_package_from_selected_project():
    resolved = bootstrap_project(PROJECT_ROOT)

    assert resolved == PROJECT_ROOT.resolve()
    assert resolved in Path(gas_portfolio_rl.__file__).resolve().parents
