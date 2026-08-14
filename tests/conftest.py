from pathlib import Path
import sys


# Ensure tests can import the top-level src package across IDE/task shells.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
