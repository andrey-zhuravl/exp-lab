import os
import subprocess
from pathlib import Path


def test_smoke():
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    subprocess.check_call(
        ["python", "-m", "exp_lab.cli", "run", "-m", "experiments/a1.quickstart.yaml"],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert Path("out/a1/report.md").exists(), "report.md not found"
