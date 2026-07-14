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
All arms write into one runs/<RUN>/ directory; bump RUN after ANY change
that invalidates old fits (code, caches, arms) — a fresh directory means
resume can never cross a code change, which is the mechanism behind the
2026-07-14 stale-row contamination.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# bump on any change that invalidates old fits; never reuse a run dir
# across code or cache changes
RUN = "2026-07-14-echo-fix"

# (out_name, extra run_flow_solver.py args) — one arm per elicitation
# protocol, same logit solver config across all of them. The cache files
# come from run_protocol_pilot.sh (committed + pushed before running this).
PROTOCOLS = ["baseline", "v1", "v1x2", "v1_fermi", "v1_spread"]
ARMS = [
    (f"pred_{p}.json",
     ["--cache", str(HERE / "caches" / p / f"llm_cache_{p}.json"),
      "--prob-penalty", "logit"])
    for p in PROTOCOLS
]
# collapse_repeats is now the solver default (duplicates count once,
# disagreement widens). Keep one ablation arm where v1x2's repeats are fed
# through raw — k penalties per quantity, sqrt(k) sharpening — so the cost
# of NOT collapsing stays measurable.
ARMS.append(
    ("pred_v1x2_nocollapse.json",
     ["--cache", str(HERE / "caches" / "v1x2" / "llm_cache_v1x2.json"),
      "--prob-penalty", "logit", "--no-collapse"]))


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    extra = sys.argv[1:]  # forwarded to every arm, e.g. --shard 0/2 --limit 5
    suffix = ""
    if "--shard" in extra:  # per-session out files; merge shards afterwards
        i = extra[extra.index("--shard") + 1].split("/")[0]
        suffix = f"_shard{i}"
    run_dir = HERE / "runs" / RUN
    outs = []
    for out_name, args in ARMS:
        cache = Path(args[args.index("--cache") + 1])
        if not cache.exists():
            print(f"skip {out_name}: no cache at {cache}", flush=True)
            continue
        out = run_dir / out_name.replace(".json", f"{suffix}.json")
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
