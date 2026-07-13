"""Colab payload: run the current pilot arms + diagnostics.

The Colab notebook never changes — it just resyncs the repo and runs this.
Edit ARMS here (locally), `git commit -am wip && git push`, rerun the
resync + payload cells.

Each arm is resume-safe: rerunning skips entries already in its --out file.
Delete the arm's json (or bump its name) after a change that invalidates
old fits — stale rows are the classic confound.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (out_name, extra run_flow_solver.py args) — equal steps across arms.
ARMS = [
    ("arm_abs.json", ["--prob-penalty", "abs"]),
    ("arm_logit.json", ["--prob-penalty", "logit"]),
    ("arm_logit_robust.json", ["--prob-penalty", "logit", "--robust"]),
]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    extra = sys.argv[1:]  # forwarded to every arm, e.g. --shard 0/2 --limit 5
    suffix = ""
    if "--shard" in extra:  # per-session out files; merge shards afterwards
        i = extra[extra.index("--shard") + 1].split("/")[0]
        suffix = f"_shard{i}"
    outs = []
    for out_name, args in ARMS:
        out = HERE / out_name.replace(".json", f"{suffix}.json")
        outs.append(out)
        run([sys.executable, str(HERE / "run_flow_solver.py"),
             "--out", str(out), *args, *extra])
    done = [o for o in outs if o.exists()]
    if not suffix and done:  # diagnostics only make sense on complete files
        run([sys.executable, str(HERE / "pilot_diagnostics.py"),
             "--predictions", *(str(o) for o in done)])


if __name__ == "__main__":
    main()
