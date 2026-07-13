"""Colab payload: benchmark each elicitation protocol through the solver.

The Colab notebook never changes — it just resyncs the repo and runs this.
Edit ARMS here (locally), `git commit -am wip && git push`, rerun the
resync + payload cells.

Elicitation (run_protocol_pilot.sh) runs locally and produces one cache per
protocol; this fits each cache through the winning (logit) solver arm on
GPU and prints one diagnostics table. Each arm differs ONLY in its
elicitation cache — same solver config — so the comparison isolates the
multi-pass protocol.

Each arm is resume-safe: rerunning skips entries already in its --out file.
Delete the arm's json (or bump its name) after a change that invalidates
old fits — stale rows are the classic confound.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (out_name, extra run_flow_solver.py args) — one arm per elicitation
# protocol, same logit solver config across all of them. The cache files
# come from run_protocol_pilot.sh (committed + pushed before running this).
PROTOCOLS = ["baseline", "v1", "v1x2", "v1_fermi"]
ARMS = [
    (f"pred_{p}.json", ["--cache", str(HERE / f"llm_cache_{p}.json"),
                        "--prob-penalty", "logit"])
    for p in PROTOCOLS
]
# v1x2's whole point is repeated fills. Plain, k repeats just sharpen the
# belief by sqrt(k) whether or not they agree (anti-calibration); the
# collapsed arm folds each repeat group into one estimate with a
# spread-derived sd, so disagreement widens instead. Compare the two.
ARMS.append(
    ("pred_v1x2_collapsed.json", ["--cache", str(HERE / "llm_cache_v1x2.json"),
                                  "--prob-penalty", "logit",
                                  "--collapse-repeats"]))


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
        # --common: score every arm on the keys fit in ALL of them, so a
        # protocol that failed on different entries doesn't skew the compare
        run([sys.executable, str(HERE / "pilot_diagnostics.py"),
             "--common", "--predictions", *(str(o) for o in done)])


if __name__ == "__main__":
    main()
