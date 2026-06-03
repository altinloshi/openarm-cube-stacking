"""Render a human-readable report from an OpenArm cube-stacking ``results.json``.

Usage::

    # Explicit file:
    python scripts/eval/make_eval_report.py --results logs/eval/openarm_cube_stack/<ts>/results.json

    # Latest results under logs/eval/openarm_cube_stack/:
    python scripts/eval/make_eval_report.py

Writes a ``report.md`` next to the results file (unless ``--out`` is given) and
prints the report to stdout. This script has no Isaac Lab dependency.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DEFAULT_EVAL_DIR = os.path.join(_PROJECT_ROOT, "logs", "eval", "openarm_cube_stack")


def _find_latest_results() -> str | None:
    matches = glob.glob(os.path.join(_DEFAULT_EVAL_DIR, "*", "results.json"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def build_report(results: dict) -> str:
    per_cube = results.get("per_cube_success_rate", [])
    per_cube_rows = "\n".join(
        f"| cube {i} | {_fmt_pct(p)} |" for i, p in enumerate(per_cube)
    )

    lines = [
        f"# OpenArm Cube Stacking — Evaluation Report",
        "",
        f"- **Task**: `{results.get('task', 'n/a')}`",
        f"- **Environments**: {results.get('num_envs', 'n/a')}",
        f"- **Episodes**: {results.get('episodes', 'n/a')}",
        "",
        "## Tournament metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Full-stack success rate | {_fmt_pct(results.get('full_stack_success_rate', 0.0))} |",
        f"| Average cubes stacked | {results.get('average_cubes_stacked', 0.0):.3f} |",
        f"| Cube drop rate | {_fmt_pct(results.get('cube_drop_rate', 0.0))} |",
        f"| Stack collapse rate | {_fmt_pct(results.get('stack_collapse_rate', 0.0))} |",
        f"| Mean final stack error (m) | {results.get('mean_final_stack_error', 0.0):.4f} |",
        f"| Mean episode length (s) | {results.get('mean_episode_length_s', 0.0):.2f} |",
        f"| Mean grasp retries | {results.get('mean_grasp_retries', 0.0):.3f} |",
        f"| Timeout rate | {_fmt_pct(results.get('timeout_rate', 0.0))} |",
        "",
        "## Per-cube success rate",
        "",
        "| Cube | Success rate |",
        "| --- | --- |",
        per_cube_rows,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an OpenArm cube-stacking evaluation report.")
    parser.add_argument("--results", type=str, default=None, help="Path to results.json (default: latest).")
    parser.add_argument("--out", type=str, default=None, help="Output report path (default: report.md beside results).")
    args = parser.parse_args()

    results_path = args.results or _find_latest_results()
    if results_path is None or not os.path.isfile(results_path):
        raise SystemExit(
            "No results.json found. Run scripts/eval/evaluate_stack.py first or pass --results."
        )

    with open(results_path) as f:
        results = json.load(f)

    report = build_report(results)
    out_path = args.out or os.path.join(os.path.dirname(results_path), "report.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n[INFO] Wrote report to: {out_path}")


if __name__ == "__main__":
    main()
